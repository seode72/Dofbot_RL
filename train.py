# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import datetime
from collections import deque
import numpy as np

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# argparse
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train DOFBOT with SAC & RFCL Curriculum.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Dofbot-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_timesteps", type=int, default=3000000,
                    help="Total environment steps for training.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--resume_run_name", type=str, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--start_stage", type=int, default=None, choices=[1, 2],
                    help="Force start stage on resume (1 or 2). Auto-detected if omitted.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# -----------------------------------------------------------------------------
# launch app
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# imports after app launch
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp

import dofbot_task
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from dofbot_task.agent.sac import SAC, SAC_DEFAULT_CONFIG, RandomMemory
from dofbot_task.agent.replay_dataset import ReplayDataset

from models.policy import PolicyModel
from models.critic import CriticModel
from models.models_cfg import PolicyModelCfg, CriticModelCfg
from checkpoint_tools.checkpoint import find_latest_checkpoint, load_checkpoint
import h5py

from mdp.reward import (
    finger_center_to_cube_distance,
    reward_reach_cube_exp,
    reward_per_finger_distance,
    reward_finger_closing,
    reward_close_approach,
    reward_single_finger_contact,
    reward_dual_finger_contact,
)
from torch.utils.tensorboard import SummaryWriter

# Import our PyTorch RFCL buffers
from custom_reverse_curriculum import ReverseCurriculumManager
from custom_forward_curriculum import ForwardCurriculumManager

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
GRASP_GOAL_JOINTS = [0.0, -0.523599, -1.18682, -0.994838, 0.0, 0.174533, -0.174533]

# 특정 env들의 로봇 joint state를 강제로 원하는 값으로 세팅
def set_robot_joints_per_env(
    base_env,
    env_ids: torch.Tensor,
    joint_pos: torch.Tensor,   # [n, 7]
    joint_vel: torch.Tensor,   # [n, 7]
    joint_ids: list,
    device: str,
) -> None:
    """지정된 env들의 로봇 관절 위치 및 속도 설정."""
    if env_ids.numel() == 0:
        return
    robot = base_env.scene["robot"]
    robot.write_joint_state_to_sim(joint_pos.to(device), joint_vel.to(device), joint_ids=joint_ids, env_ids=env_ids)

# 특정 env들의 큐브 pose/vel을 직접 시뮬레이터에 사용
def set_cube_state_per_env(
    base_env,
    env_ids: torch.Tensor,
    cube_state: torch.Tensor,  # [n, 13] (local_pos 3 + quat 4 + lin_vel 3 + ang_vel 3)
    device: str,
) -> None:
    """지정된 env들의 큐브 포즈 및 속도 설정. (환경 원점 오프셋 반영)"""
    if env_ids.numel() == 0:
        return
    cube = base_env.scene["cube"]
    
    # 데모 데이터는 local_pos이므로 환경의 원점을 더해줌
    local_pos = cube_state[:, 0:3].to(device)
    origins = base_env.scene.env_origins[env_ids].to(device)
    world_pos = local_pos + origins
    
    quat = cube_state[:, 3:7].to(device)
    
    # 디버그: 쿼터니언이 모두 0이면 비정상
    if torch.all(quat == 0):
        print(f"[ERROR] All-zero quaternion detected in environment {env_ids}!")
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(quat.shape[0], 1)
    
    # USD Orthonormalize 오류 방지를 위해 쿼터니언 정규화 강제
    quat = torch.nn.functional.normalize(quat, dim=-1)
    
    lin_vel = cube_state[:, 7:10].to(device)
    ang_vel = cube_state[:, 10:13].to(device)
    
    # Pose와 Velocity 각각 반영
    cube.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)


def extract_policy_obs(obs_dict) -> torch.Tensor:
    policy_obs = obs_dict["policy"]
    obs_parts = [
        # policy_obs["joint_effort"],          # [N, 7]
        policy_obs["joint_pos"],             # [N, 7]
        policy_obs["joint_vel"],             # [N, 7]
        policy_obs["cube_pos"],              # [N, 3]
        # policy_obs["left_finger_pos"],       # [N, 3]
        # policy_obs["right_finger_pos"],      # [N, 3]
        # policy_obs["left_finger_contact"],   # [N, 1]
        # policy_obs["right_finger_contact"],  # [N, 1]
        # policy_obs["cube_to_finger_vec"],    # [N, 3]
    ]
    return torch.cat(obs_parts, dim=-1)


def compute_reward_terms(env) -> dict[str, torch.Tensor]:
    """Reward Term 반환 (Raw 값)"""
    return {
        "reach_cube":            reward_reach_cube_exp(env, temperature=0.08),
        "per_finger_dist":       reward_per_finger_distance(env, temperature=0.10),
        "finger_closing":        reward_finger_closing(env, activation_dist=0.15),
        "close_approach":        reward_close_approach(env),
        "single_finger_contact": reward_single_finger_contact(env),
        "dual_finger_contact":   reward_dual_finger_contact(env),
        "action_rate_penalty":   mdp.action_rate_l2(env),
        "joint_vel_penalty":     mdp.joint_vel_l2(env),
    }

# seed_offline_memory_from_h5 제거 — ReplayDataset 이 직접 H5를 로드하므로 불필요



# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs_dict, _ = env.reset()
    obs = extract_policy_obs(obs_dict).to(args_cli.device)

    obs_dim    = obs.shape[-1]
    action_dim = base_env.action_manager.total_action_dim
    num_envs   = base_env.num_envs

    print(f"[INFO] obs_dim={obs_dim}, action_dim={action_dim}, num_envs={num_envs}")
    print(f"[INFO] body_names: {base_env.scene['robot'].body_names}")

    # SAC 설정
    sac_cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    sac_cfg["batch_size"] = 256           # 원본 rfcl: 256
    sac_cfg["gradient_steps"] = 80        # 원본 rfcl: grad_updates_per_step=80
    sac_cfg["actor_learning_rate"] = 3e-4
    sac_cfg["critic_learning_rate"] = 3e-4
    sac_cfg["entropy_learning_rate"] = 3e-4
    sac_cfg["random_timesteps"] = 2000
    sac_cfg["learning_starts"] = 5000
    sac_cfg["discount_factor"] = 0.9      # 원본 rfcl: 0.9 (sparse reward에 적합)
    sac_cfg["experiment"]["directory"] = "logs"
    sac_cfg["experiment"]["experiment_name"] = "sac_rfcl_0412"
    sac_cfg["experiment"]["write_interval"] = 1000
    sac_cfg["experiment"]["checkpoint_interval"] = 5000

    # KST Timeline update
    kst = datetime.timezone(datetime.timedelta(hours=9))
    timestamp = datetime.datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
    exp_base = os.path.join(
        sac_cfg["experiment"]["directory"],
        sac_cfg["experiment"]["experiment_name"],
    )
    os.makedirs(exp_base, exist_ok=True)
    checkpoint_path = args_cli.checkpoint
    resume_root = None

    # --checkpoint 또는 --resume_run_name 이 지정되면 자동으로 resume 모드
    auto_resume = args_cli.resume or (checkpoint_path is not None) or (args_cli.resume_run_name is not None)
    if auto_resume:
        if checkpoint_path is not None:
            resume_root = os.path.dirname(checkpoint_path)
        else:
            resume_root = os.path.join(exp_base, args_cli.resume_run_name)
            checkpoint_path = find_latest_checkpoint(resume_root)

        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint not found. resume_root={resume_root}")
        args_cli.resume = True  # 이후 분기에서 resume 경로 타도록 강제
        print(f"[INFO] Auto-resume enabled. resume_root={resume_root}, checkpoint={checkpoint_path}")

    if args_cli.resume and resume_root is not None:
        current_run_name = os.path.basename(resume_root.rstrip("/"))
        current_run_root = resume_root
    else:
        current_run_name = timestamp
        current_run_root = os.path.join(exp_base, current_run_name)
    os.makedirs(current_run_root, exist_ok=True)

    tb_dir = os.path.join(current_run_root, "TB")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"[INFO] TensorBoard logs: {tb_dir}")
    with open(os.path.join(current_run_root, "resume_log.txt"), "a") as f:
        f.write(f"[{timestamp}] resume={args_cli.resume} ckpt={checkpoint_path}\n")

    # 모델 / Agent 생성
    print("[DEBUG] Step 1: Creating Models (Policy, 10 Critics, 10 Target Critics)...")
    NUM_QS = 10  # 원본 rfcl: num_qs=10
    policy = PolicyModel(obs_dim, action_dim, PolicyModelCfg(), args_cli.device)
    critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), args_cli.device) for _ in range(NUM_QS)]
    target_critics = [CriticModel(obs_dim, action_dim, CriticModelCfg(), args_cli.device) for _ in range(NUM_QS)]

    # 타겟 네트워크 초기 동기화 (SAC.__init__에서도 수행하지만 명시적으로)
    for critic, target in zip(critics, target_critics):
        target.load_state_dict(critic.state_dict())

    # Online memory (skrl RandomMemory) -- Online Replay Buffer
    memory = RandomMemory(
        memory_size=1000000,
        num_envs=num_envs,
        device=args_cli.device
    )
    # Offline buffer: 원본 rfcl 스타일 ReplayDataset (단순 dict of tensors)
    h5_path = "scripts/dofbot_0412/demos/demos.h5"
    print(f"[DEBUG] Step 2: Loading ReplayDataset from {h5_path}...")
    offline_buffer = ReplayDataset(h5_path, device=args_cli.device)
    print(f"[DEBUG] Step 2: ReplayDataset loading finished.")

    models_dict = {"policy": policy}
    for i, c in enumerate(critics):
        models_dict[f"critic_{i+1}"] = c
    for i, tc in enumerate(target_critics):
        models_dict[f"target_critic_{i+1}"] = tc

    print("[DEBUG] Step 3: Creating SAC Agent...")
    agent = SAC(
        models=models_dict,
        memory=memory,
        observation_space=obs_dim,
        action_space=action_dim,
        device=args_cli.device,
        cfg=sac_cfg,
    )
    # 원본 rfcl 스타일: algo.offline_buffer = demo_replay_dataset
    agent.offline_buffer = offline_buffer
    agent.init()

    # Checkpoint 복원
    total_step = 0
    completed_eps = 0

    resumed_stage = None
    if checkpoint_path is not None:
        print(f"[INFO] Resuming from: {checkpoint_path}")
        ckpt_result = load_checkpoint(agent, checkpoint_path, args_cli.device)
        if isinstance(ckpt_result, tuple):
            if len(ckpt_result) == 3:
                total_step, completed_eps, resumed_stage = ckpt_result
            else:
                total_step, completed_eps = ckpt_result
        else:
            total_step = ckpt_result

        print(f"[INFO] Resumed at total_step={total_step}, completed_eps={completed_eps}")
    else:
        print("[INFO] Starting fresh training.")

    # ReplayDataset 은 이미 위에서 생성되어 agent.offline_buffer 에 할당됨

    # --- 인터페이스 자가 진단 (Dry-run) ---
    print("[INFO] Performing interface dry-run...")
    try:
        test_obs = torch.randn(1, obs_dim, device=args_cli.device)
        test_output = agent.act(test_obs, 0, 0)
        print(f"[INFO] Dry-run success! act() returned {len(test_output)} values.")
    except Exception as e:
        print(f"[CRITICAL] Interface dry-run failed: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return
    # -------------------------------------

    ep_reward_sum = torch.zeros(num_envs, device=args_cli.device)
    reward_term_names = [
        "reach_cube", "per_finger_dist", "finger_closing", "close_approach",
        "single_finger_contact", "dual_finger_contact",
        "action_rate_penalty", "joint_vel_penalty",
    ]
    ep_raw_sum = {k: torch.zeros(num_envs, device=args_cli.device) for k in reward_term_names}
    gamma = sac_cfg["discount_factor"]
    ep_discounted_return = torch.zeros(num_envs, device=args_cli.device)
    ep_step_count = torch.zeros(num_envs, device=args_cli.device)
    ep_success_steps = torch.zeros(num_envs, dtype=torch.int32, device=args_cli.device)

    term_count_total = 0
    trunc_count_total = 0
    recent_ep_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    total_timesteps = args_cli.total_timesteps
    best_succ_rate = -1.0
    if args_cli.resume and resume_root is not None:
        best_ckpt_prev = os.path.join(resume_root, "best_agent.pt")
        if os.path.exists(best_ckpt_prev):
            try:
                prev = torch.load(best_ckpt_prev, map_location="cpu")
                if "best_succ_rate" in prev:
                    best_succ_rate = float(prev["best_succ_rate"])
                    print(f"[INFO] Restored best_succ_rate={best_succ_rate:.2f} from {best_ckpt_prev}")
                else:
                    print(f"[WARN] best_agent.pt 존재하지만 best_succ_rate 미기록 → -1.0으로 시작")
            except Exception as e:
                print(f"[WARN] best_agent.pt 로드 실패: {e}")
    stage2_start_eps = -1
    solved_frac_stable_steps = 0
    last_stable_check_step = 0
    STABLE_THRESHOLD = 10000 # 1만 env step 유지 조건

    # RFCL Buffer 초기화
    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4",
        "Wrist_Twist_RevoluteJoint",
        "Finger_Left_01_RevoluteJoint", "Finger_Right_01_RevoluteJoint",
    ]
    joint_ids = []
    robot = base_env.scene["robot"]
    for name in arm_joint_names:
        jid = robot.find_joints(name)[0]
        if isinstance(jid, (list, tuple)):
            jid = jid[0]
        if isinstance(jid, torch.Tensor):
            jid = jid.item()
        joint_ids.append(int(jid))

    h5_path = "scripts/dofbot_0412/demos/demos.h5"
    reverse_manager = ReverseCurriculumManager(
        h5_path=h5_path,
        reverse_step_size=8,
        per_demo_buffer_size=24, # 24로 상향
        threshold=0.9
    )

    if args_cli.resume and resume_root is not None:
        rcg_buffer_path = os.path.join(resume_root, f"rcg_buffer_step{total_step}.pt")
    else:
        rcg_buffer_path = os.path.join(current_run_root, f"rcg_buffer_step{total_step}.pt")

    if total_step > 0 and os.path.exists(rcg_buffer_path):
        rcg_state = torch.load(rcg_buffer_path, map_location="cpu")
        reverse_manager.load_state(rcg_state)
        print("[RCG] Loaded RFCL state.")

    forward_manager = ForwardCurriculumManager(num_seeds=1000, num_envs=num_envs)

    # ---- Stage 자동 감지 + stage2 offline buffer 복원 ----
    current_stage = 1
    import glob
    search_roots = []
    if resume_root is not None:
        search_roots.append(resume_root)
    if checkpoint_path is not None:
        search_roots.append(os.path.dirname(checkpoint_path))
    # 중복 제거
    search_roots = list(dict.fromkeys(search_roots))

    stage2_files = []
    for r in search_roots:
        found = sorted(glob.glob(os.path.join(r, "offline_buffer_stage2_*.pt")))
        if found:
            stage2_files = found
            break
    detected_stage2 = len(stage2_files) > 0 or resumed_stage == 2

    print(f"[STAGE-DETECT] search_roots={search_roots}")
    print(f"[STAGE-DETECT] stage2_files={stage2_files}")
    print(f"[STAGE-DETECT] resumed_stage={resumed_stage}, --start_stage={args_cli.start_stage}")

    forced_stage = args_cli.start_stage
    if forced_stage is not None:
        current_stage = forced_stage
    elif detected_stage2:
        current_stage = 2

    if current_stage == 2:
        if stage2_files:
            buf_path = stage2_files[-1]
            state = torch.load(buf_path, map_location=args_cli.device)
            agent.offline_buffer.load_state_dict(state)
            print(f"[INFO] Stage2 offline buffer loaded: {buf_path} ({len(agent.offline_buffer)} transitions)")
        else:
            print(f"[WARN] start_stage=2 requested but no offline_buffer_stage2_*.pt found in {search_roots}")
        print(f"[INFO] Resuming at STAGE 2 (reverse curriculum skipped).")

    # 각 환경별 Tracker
    ep_demo_ids = torch.zeros(num_envs, dtype=torch.long, device=args_cli.device)
    ep_steps_back = torch.zeros(num_envs, dtype=torch.long, device=args_cli.device)
    ep_max_steps = torch.zeros(num_envs, dtype=torch.long, device=args_cli.device)

    # 초기 에피소드 할당 (stage1 전용 — stage2 resume 시 reverse_manager 사용 안 함)
    if current_stage == 1:
        for env_id in range(num_envs):
            j_pos, j_vel, c_state, p_act, d_id, s_back = reverse_manager.generate_next(args_cli.device)
            ep_demo_ids[env_id] = d_id
            ep_steps_back[env_id] = s_back
            # dynamic timeout — steps_back의 6배 + 기본 여유 (전체 데모 시 ~300스텝)
            ep_max_steps[env_id] = 200 + (s_back) * 6

            env_ids_t = torch.tensor([env_id], device=args_cli.device)
            set_robot_joints_per_env(base_env, env_ids_t, j_pos.unsqueeze(0), j_vel.unsqueeze(0), joint_ids, args_cli.device)
            set_cube_state_per_env(base_env, env_ids_t, c_state.unsqueeze(0), args_cli.device)
            # Action 복원 (연속성 보장)
            base_env.action_manager.action[env_id] = p_act.to(args_cli.device)

        base_env.scene.write_data_to_sim()
    else:
        # Stage2: 환경 기본 reset 분포 사용
        ep_max_steps[:] = 300
    obs_dict = base_env.observation_manager.compute()
    obs = extract_policy_obs(obs_dict).to(args_cli.device)

    ep_init_dist = finger_center_to_cube_distance(base_env).view(num_envs).clone()
    GRASP_GOAL_TENSOR = torch.tensor(GRASP_GOAL_JOINTS, dtype=torch.float32, device=args_cli.device)
    JOINT_THRESHOLD = 0.2

    print(f"[INFO] Start training. total_timesteps={total_timesteps}")

    while simulation_app.is_running() and total_step < total_timesteps:
        skip_update = False

        with torch.no_grad():
            agent.pre_interaction(timestep=total_step, timesteps=total_timesteps)
            actions, _, _ = agent.act(
                states=obs,
                timestep=total_step,
                timesteps=total_timesteps,
            )

        next_obs_dict, rewards, terminated, truncated, extras = env.step(actions)
        next_obs = extract_policy_obs(next_obs_dict).to(args_cli.device).detach()

        rewards_1d    = rewards.view(num_envs).detach()
        terminated_1d = terminated.view(num_envs).detach()

        # Compute dual contact for success logic and Sparse Reward
        dual_contact = reward_dual_finger_contact(base_env).view(num_envs).detach()
        is_current_success = dual_contact > 0.0
        ep_success_steps = torch.where(is_current_success, ep_success_steps + 1, 0)
        
        # Early Terminate condition
        grace_steps = 15
        early_term_1d = ep_success_steps >= grace_steps
        terminated_1d = terminated_1d | early_term_1d
        
        # Override with Sparse Reward
        # We also keep a small proportion of the dense reward just to smooth learning if needed,
        # but dominantly it's a sparse 1.0 for holding the cube.
        # Custom reward = 1.0 when successful hold, else 0.0 + 0.01 * original_dense
        # 즉, Dense Reward는 1%만 Mix
        custom_rewards_1d = is_current_success.float() + (0.01 * rewards_1d)

        # Dynamic trunc calculation
        ep_step_count += 1
        dynamic_trunc_1d = (ep_step_count >= ep_max_steps).detach()
        truncated_1d = truncated.view(num_envs).detach() | dynamic_trunc_1d
        
        dones_1d = terminated_1d | truncated_1d

        ep_reward_sum  += custom_rewards_1d
        raw_terms = compute_reward_terms(base_env)
        for k in reward_term_names:
            ep_raw_sum[k] += raw_terms[k].view(num_envs).detach()
        ep_discounted_return = ep_discounted_return * gamma + custom_rewards_1d

        curr_dist = finger_center_to_cube_distance(base_env).view(num_envs).detach()

        # agent 에 기록. SAC는 buffer에 바로 들어감
        agent.record_transition(
            states=obs,
            actions=actions,
            rewards=custom_rewards_1d.view(num_envs, 1),
            next_states=next_obs,
            terminated=terminated_1d.view(num_envs, 1),
            truncated=truncated_1d.view(num_envs, 1),
            infos=extras,
            timestep=total_step,
            timesteps=total_timesteps,
        )
        
        done_idx = dones_1d.nonzero(as_tuple=False).squeeze(-1)

        if done_idx.numel() > 0:
            k = done_idx.numel()
            completed_eps += k

            n_term  = terminated_1d[done_idx].sum().item()
            n_trunc = truncated_1d[done_idx].sum().item()
            term_count_total  += n_term
            trunc_count_total += n_trunc

            ep_rew_mean  = ep_reward_sum[done_idx].mean().item()
            recent_ep_rewards.append(ep_rew_mean)
            # Success logic follows collect_demos.py (early terminated due to grace_steps)
            is_success_batch = (ep_success_steps[done_idx] >= grace_steps)
            recent_successes.append(is_success_batch.float().mean().item())
            
            rew_moving_avg = sum(recent_ep_rewards) / len(recent_ep_rewards)
            succ_moving_avg = sum(recent_successes) / len(recent_successes) * 100.0

            if current_stage == 1:
                rcg_state_log = reverse_manager.log_state()
                current_solved_frac = rcg_state_log.get("solved_frac", 0.0)
                details = rcg_state_log.get("details", {})
                prog_strs = []
                for k, v in details.items():
                    if k.startswith("demo_"):
                        prog_strs.append(f"D{k.split('_')[1]}:{int((1.0 - v)*100)}%")
                prog_str = " ".join(prog_strs)

                print(
                    f"[EP {completed_eps:>6}] "
                    f"step={total_step:>8} | "
                    f"rew={ep_rew_mean:>8.4f} | "
                    f"mov_avg={rew_moving_avg:>8.4f} | "
                    f"succ={succ_moving_avg:>5.1f}% | "
                    f"sol={current_solved_frac:>4.2f} | "
                    f"[{prog_str}]"
                )
            else:
                current_solved_frac = 1.0
                print(
                    f"[STAGE2 EP {completed_eps:>6}] "
                    f"step={total_step:>8} | "
                    f"rew={ep_rew_mean:>8.4f} | "
                    f"mov_avg={rew_moving_avg:>8.4f} | "
                    f"succ={succ_moving_avg:>5.1f}%"
                )

            # --- Best Agent Saving for Stage 2 ---
            if current_stage == 2:
                if stage2_start_eps == -1:
                    stage2_start_eps = completed_eps
                
                # Stage 2 진입 후 최소 50 에피소드 이상 경과 시점부터 Best 측정 (성공률 안정화 대기)
                if (completed_eps - stage2_start_eps) > 50:
                    if succ_moving_avg > best_succ_rate:
                        best_succ_rate = succ_moving_avg
                        from checkpoint_tools.checkpoint import save_checkpoint
                        best_ckpt_path = os.path.join(current_run_root, "best_agent.pt")
                        save_checkpoint(agent, best_ckpt_path, total_step, completed_eps, current_stage=current_stage, best_succ_rate=best_succ_rate)
                        print(f"[@] New Best Agent Saved! Success Rate: {best_succ_rate:.1f}% (Step: {total_step})")
            # -------------------------------------

            if completed_eps % 100 == 0:
                tag_prefix = "reverse" if current_stage == 1 else "forward"
                writer.add_scalar(f"{tag_prefix}/reward_moving_avg_100", rew_moving_avg, total_step)
                writer.add_scalar(f"{tag_prefix}/success_rate_100", succ_moving_avg, total_step)
                writer.add_scalar("rcg/solved_frac", current_solved_frac, total_step)
                disc_ret_mean = ep_discounted_return[done_idx].mean().item()
                writer.add_scalar(f"{tag_prefix}/discounted_return", disc_ret_mean, total_step)
                for key in reward_term_names:
                    raw_mean = ep_raw_sum[key][done_idx].mean().item()
                    writer.add_scalar(f"reward_raw_{tag_prefix}/{key}", raw_mean, total_step)
                writer.flush()

            ep_reward_sum[done_idx] = 0.0
            ep_discounted_return[done_idx] = 0.0
            ep_step_count[done_idx] = 0.0
            ep_success_steps[done_idx] = 0
            for key in reward_term_names:
                ep_raw_sum[key][done_idx] = 0.0

            # RFCL / RCG Record & Reset
            # Note: new_joints_list 폐기, 즉시 주입 방식으로 통합
            
            for i, env_id in enumerate(done_idx.tolist()):
                success = is_success_batch[i].item()
                
                if current_stage == 1:
                    # Stage 1: Reverse Curriculum
                    d_id = ep_demo_ids[env_id].item()
                    s_back = ep_steps_back[env_id].item()
                    reverse_manager.record(d_id, s_back, success)
                    
                    j_pos, j_vel, c_state, p_act, next_d_id, next_s_back = reverse_manager.generate_next(args_cli.device)
                    ep_demo_ids[env_id] = next_d_id
                    ep_steps_back[env_id] = next_s_back
                    ep_max_steps[env_id] = 200 + (next_s_back) * 6
                    
                    env_ids_t = torch.tensor([env_id], device=args_cli.device)
                    set_robot_joints_per_env(base_env, env_ids_t, j_pos.unsqueeze(0), j_vel.unsqueeze(0), joint_ids, args_cli.device)
                    set_cube_state_per_env(base_env, env_ids_t, c_state.unsqueeze(0), args_cli.device)
                    
                    # Action 복원 (연속성 보장)
                    base_env.action_manager.action[env_id] = p_act.to(args_cli.device)
                else:
                    # Stage 2: Forward Curriculum
                    seed_val = ep_demo_ids[env_id].item()
                    forward_manager.record_episode(seed_val, ep_reward_sum[env_id].item(), success)

                    next_seeds, _ = forward_manager.sample_seeds(1)
                    ep_demo_ids[env_id] = next_seeds[0]
                    ep_max_steps[env_id] = 200 # Fixed horizon for Stage 2

                    # 원본 rfcl: seed 기반 deterministic 초기상태 (seed가 같으면 같은 초기상태)
                    seed_rng = np.random.RandomState(int(next_seeds[0]))
                    base_j_pos = np.array([0.0, 0.0, -1.1, -1.5, 0.0, 0.0, 0.0])
                    j_perturbation = seed_rng.uniform(-0.05, 0.05, size=7)
                    # 손가락은 perturbation 최소화
                    j_perturbation[5:] *= 0.3
                    init_j_pos = torch.tensor(base_j_pos + j_perturbation, dtype=torch.float32, device=args_cli.device)
                    init_j_vel = torch.zeros(7, dtype=torch.float32, device=args_cli.device)

                    # 큐브 위치도 seed 기반 랜덤화
                    cube_x = seed_rng.uniform(-0.03, 0.03)
                    cube_y = seed_rng.uniform(0.17, 0.23)
                    cube_z = 0.03
                    init_cube_state = torch.tensor(
                        [cube_x, cube_y, cube_z,  1.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
                        dtype=torch.float32, device=args_cli.device,
                    )

                    env_ids_t = torch.tensor([env_id], device=args_cli.device)
                    set_robot_joints_per_env(base_env, env_ids_t, init_j_pos.unsqueeze(0), init_j_vel.unsqueeze(0), joint_ids, args_cli.device)
                    set_cube_state_per_env(base_env, env_ids_t, init_cube_state.unsqueeze(0), args_cli.device)
            
            if current_stage == 1:
                rcg_state_log = reverse_manager.log_state()
                solved_frac = rcg_state_log.get("solved_frac", 0.0)
                
                # 안정성 체크: 0.95 이상 유지 (실제 env step 기반)
                if solved_frac > 0.95:
                    solved_frac_stable_steps += (total_step - last_stable_check_step) * num_envs
                else:
                    solved_frac_stable_steps = 0
                last_stable_check_step = total_step

                if solved_frac > 0.95 and solved_frac_stable_steps > STABLE_THRESHOLD:
                    print(f"==========> [RFCL Stage 1 Complete] Stage 2 transition. total_step={total_step} <==========")
                    current_stage = 2

                    # 1) Online buffer → Offline buffer 변환
                    print(f"[INFO] Converting Stage 1 Online buffer ({len(agent.memory)} samples) to Stage 2 Offline buffer")
                    agent.offline_buffer = ReplayDataset.from_online_buffer(agent.memory, device=args_cli.device)

                    # 2) Offline buffer 디스크 저장 (크래시 복구용)
                    offline_save_path = os.path.join(current_run_root, f"offline_buffer_stage2_{total_step}.pt")
                    torch.save(agent.offline_buffer.state_dict(), offline_save_path)
                    print(f"[INFO] Offline buffer saved: {offline_save_path} ({len(agent.offline_buffer)} transitions)")

                    # 3) Online 버퍼 리셋
                    agent.memory = RandomMemory(
                        memory_size=1000000,
                        num_envs=num_envs,
                        device=args_cli.device
                    )
                    agent.memory.create_tensor(name="states", size=obs_dim, dtype=torch.float32)
                    agent.memory.create_tensor(name="next_states", size=obs_dim, dtype=torch.float32)
                    agent.memory.create_tensor(name="actions", size=action_dim, dtype=torch.float32)
                    agent.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
                    agent.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
                    agent.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

                    # 4) 체크포인트 저장
                    from checkpoint_tools.checkpoint import save_checkpoint
                    save_checkpoint(agent, os.path.join(current_run_root, f"stage1_final_{total_step}.pt"), total_step, completed_eps, current_stage=current_stage)

                    # 5) 이번 스텝은 학습 스킵 (빈 online buffer에서 샘플링 방지)
                    skip_update = True
            
            # 리셋 데이터 시뮬레이션 즉시 주입 완료
            base_env.scene.write_data_to_sim()
            fresh_obs_dict = base_env.observation_manager.compute()
            fresh_obs = extract_policy_obs(fresh_obs_dict).to(args_cli.device)
            next_obs[done_idx] = fresh_obs[done_idx]

            fresh_dist = finger_center_to_cube_distance(base_env).view(num_envs)
            ep_init_dist[done_idx] = fresh_dist[done_idx].clone()

            if completed_eps % 100 == 0 and current_stage == 1:
                rcg_state_log = reverse_manager.log_state()
                writer.add_scalar("rcg/mean_start_step_frac", rcg_state_log["mean_start_step_frac"], completed_eps)

        if not skip_update:
            agent.post_interaction(timestep=total_step, timesteps=total_timesteps)

        if total_step > 0 and total_step % 5000 == 0:
            from checkpoint_tools.checkpoint import save_checkpoint
            ckpt_path = os.path.join(current_run_root, f"checkpoint_{total_step}.pt")
            save_checkpoint(agent, ckpt_path, total_step, completed_eps, current_stage=current_stage)
            
            if current_stage == 1:
                rcg_buffer_path = os.path.join(current_run_root, f"rcg_buffer_step{total_step}.pt")
                torch.save(reverse_manager.save_state(), rcg_buffer_path)
            print(f"[INFO] Saved checkpoint at step {total_step}")

        obs = next_obs
        total_step += 1

    print(f"[INFO] Training finished.")
    writer.close()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()