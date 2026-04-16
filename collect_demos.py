"""
collect_demos.py
================
best_agent.pt로 성공(dual_finger_contact > 0) trajectory를 수집하고
RFCL ReplayDataset 형식(h5 + json)과 states_dataset.pkl로 저장한다.

성공 기준: reward_dual_finger_contact > 0 인 스텝이 하나라도 있으면 성공.

저장:
  demos/demos.h5   - traj_0, traj_1, ...  각각: obs[T+1,35], actions[T,7],
                                                  success[T], rewards[T], states[T+1,27]
  demos/demos.json - 에피소드 메타 (episode_id, success, reset_kwargs)
  demos/states_dataset.pkl - {demo_id: {state, seed, reset_kwargs}} (InitialStateWrapper 형식)

사용법:
    ./isaaclab.sh -p scripts/dofbot_0331/collect_demos.py \\
        --checkpoint logs/ppo_0331/checkpoints/best_agent.pt \\
        --num_demos 20 --max_episodes 300
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task",         type=str, default="Isaac-Dofbot-v0")
parser.add_argument("--checkpoint",   type=str, required=True)
parser.add_argument("--num_demos",    type=int, default=5,
                    help="수집할 성공 데모 수")
parser.add_argument("--max_episodes", type=int, default=500,
                    help="최대 시도 에피소드 수")
parser.add_argument("--out_dir",      type=str, default="demos")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--grace_steps",  type=int, default=15,
                    help="성공 감지 후 몇 스텝 더 실행한 뒤 trajectory를 자를지")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after launch ---
import copy
import json
import os
import pickle

import h5py
import numpy as np
import torch
import gymnasium as gym

import dofbot_task
from dofbot_task.agent.ppo import PPO, PPO_DEFAULT_CONFIG
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from skrl.memories.torch import Memory

from models.policy import PolicyModel
from models.value import ValueModel
from models.models_cfg import PolicyModelCfg, ValueModelCfg
from mdp.reward import reward_dual_finger_contact

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "Wrist_Twist_RevoluteJoint",
    "Finger_Left_01_RevoluteJoint",
    "Finger_Right_01_RevoluteJoint",
]


def extract_policy_obs(obs_dict) -> torch.Tensor:
    p = obs_dict["policy"]
    return torch.cat([
        # p["joint_effort"],         # [N, 7]
        p["joint_pos"],            # [N, 7]
        p["joint_vel"],            # [N, 7]
        p["cube_pos"],             # [N, 3]
        # p["left_finger_pos"],      # [N, 3]
        # p["right_finger_pos"],     # [N, 3]
        # p["left_finger_contact"],  # [N, 1]
        # p["right_finger_contact"], # [N, 1]
        # p["cube_to_finger_vec"],   # [N, 3]
    ], dim=-1)


def get_joint_ids(base_env) -> list[int]:
    ids = []
    for name in ARM_JOINT_NAMES:
        jid = base_env.scene["robot"].find_joints(name)[0]
        if isinstance(jid, (list, tuple)):
            jid = jid[0]
        if isinstance(jid, torch.Tensor):
            jid = jid.item()
        ids.append(int(jid))
    return ids


def capture_state(base_env, joint_ids: list[int]) -> np.ndarray:
    """
    현재 환경 상태를 70-dim float32 배열로 반환.

    Layout:
      [0:7]   joint_pos          - 로봇 관절 위치
      [7:14]  joint_vel          - 로봇 관절 속도
      [14:17] cube_local_pos     - cube 위치 (env_origin 기준)
      [17:21] cube_quat          - cube 방향
      [21:24] cube_lin_vel       - cube 선속도
      [24:27] cube_ang_vel       - cube 각속도
      [27:45] left_contact_hist  - 왼쪽 손가락 접촉력 history [6×3=18]
      [45:63] right_contact_hist - 오른쪽 손가락 접촉력 history [6×3=18]
      [63:70] prev_action        - 직전 action (action_rate_l2 복원용)

    RFCL 역방향 커리큘럼에서 임의 t_i 지점 복원 시 contact history와
    prev_action까지 재현하기 위해 저장한다.
    """
    robot  = base_env.scene["robot"]
    cube   = base_env.scene["cube"]
    origin = base_env.scene.env_origins[0].cpu().numpy()

    joint_pos  = robot.data.joint_pos[0, joint_ids].cpu().numpy()
    joint_vel  = robot.data.joint_vel[0, joint_ids].cpu().numpy()
    cube_pos_w = cube.data.root_pos_w[0].cpu().numpy()
    cube_quat  = cube.data.root_quat_w[0].cpu().numpy()
    cube_lv    = cube.data.root_lin_vel_w[0].cpu().numpy()
    cube_av    = cube.data.root_ang_vel_w[0].cpu().numpy()

    # contact sensor net force history: [history_length=6, 3] → flatten → [18]
    left_sensor  = base_env.scene["contact_sensor_left_finger"]
    right_sensor = base_env.scene["contact_sensor_right_finger"]
    left_hist  = left_sensor.data.net_forces_w[0].cpu().numpy().flatten()   # [18]
    right_hist = right_sensor.data.net_forces_w[0].cpu().numpy().flatten()  # [18]

    # prev_action: action_manager.action은 마지막으로 apply된 action
    # 첫 스텝 이전에는 zeros (action_manager 초기화 상태)
    try:
        prev_action = base_env.action_manager.action[0].cpu().numpy()  # [7]
    except Exception:
        prev_action = np.zeros(7, dtype=np.float32)

    return np.concatenate([
        joint_pos, joint_vel,
        cube_pos_w - origin, cube_quat,
        cube_lv, cube_av,
        left_hist, right_hist,
        prev_action,
    ]).astype(np.float32)  # [70]


def main():
    device = args_cli.device

    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device

    env      = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    obs_dict, _ = env.reset()
    obs = extract_policy_obs(obs_dict).to(device)
    obs_dim    = obs.shape[-1]
    action_dim = base_env.action_manager.total_action_dim

    joint_ids = get_joint_ids(base_env)

    # --- policy 로드 ---
    policy_m = PolicyModel(obs_dim, action_dim, PolicyModelCfg(), device)
    value_m  = ValueModel(obs_dim, 1,          ValueModelCfg(),  device)
    memory   = Memory(1, 1, device)
    ppo_cfg  = copy.deepcopy(PPO_DEFAULT_CONFIG)
    agent = PPO(
        models={"policy": policy_m, "value": value_m},
        memory=memory,
        observation_space=obs_dim,
        action_space=action_dim,
        device=device,
        cfg=ppo_cfg,
    )
    agent.init()

    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    policy_m.load_state_dict(ckpt["policy"])
    if "value" in ckpt:
        value_m.load_state_dict(ckpt["value"])
    print(f"[INFO] Loaded: {args_cli.checkpoint}")

    out_dir = os.path.join(SCRIPT_DIR, args_cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- 수집 ---
    collected: list[dict] = []
    ep_count = 0

    grace_steps   = args_cli.grace_steps  # 첫 성공 후 이 스텝만큼 더 실행 후 자름

    def _reset_ep_buffers():
        return (
            [obs[0].cpu().numpy()],
            [capture_state(base_env, joint_ids)],
            [], [],
        )

    ep_obs, ep_states, ep_actions, ep_success = _reset_ep_buffers()
    first_success_step: int | None = None  # 이번 ep에서 처음 성공한 스텝 index

    print(f"[INFO] target={args_cli.num_demos} demos, max_ep={args_cli.max_episodes}, grace_steps={grace_steps}")

    while (
        simulation_app.is_running()
        and len(collected) < args_cli.num_demos
        and ep_count < args_cli.max_episodes
    ):
        with torch.no_grad():
            actions, _, _ = agent.act(states=obs, timestep=0, timesteps=0)

        next_obs_dict, _, terminated, truncated, _ = env.step(actions)
        next_obs = extract_policy_obs(next_obs_dict).to(device)

        dual = reward_dual_finger_contact(base_env)[0].item()
        step_success = float(dual > 0.0)

        ep_actions.append(actions[0].cpu().numpy())
        ep_success.append(step_success)
        ep_states.append(capture_state(base_env, joint_ids))
        ep_obs.append(next_obs[0].cpu().numpy())

        # 처음 성공한 스텝 기록
        if step_success > 0.5 and first_success_step is None:
            first_success_step = len(ep_actions) - 1  # 0-indexed

        env_done = (terminated | truncated)[0].item()

        # 성공 후 grace_steps 만큼 지나면 강제 종료
        grace_done = (
            first_success_step is not None
            and (len(ep_actions) - 1 - first_success_step) >= grace_steps
        )

        if env_done or grace_done:
            ep_count += 1
            succeeded = first_success_step is not None
            n_success_steps = sum(1 for s in ep_success if s > 0.5)

            if succeeded:
                collected.append({
                    "obs":     np.array(ep_obs,     dtype=np.float32),  # [T+1, 35]
                    "actions": np.array(ep_actions, dtype=np.float32),  # [T,   7]
                    "success": np.array(ep_success, dtype=np.float32),  # [T]
                    "rewards": np.array(ep_success, dtype=np.float32),  # [T]  sparse
                    "states":  np.array(ep_states,  dtype=np.float32),  # [T+1, 27]
                })
                cut_reason = "grace" if grace_done else "env_done"
                print(
                    f"[EP {ep_count:>4}] SUCCESS {len(collected):>3}/{args_cli.num_demos}"
                    f"  len={len(ep_actions)}  success_steps={n_success_steps}"
                    f"  cut={cut_reason}"
                )
            else:
                print(f"[EP {ep_count:>4}] fail  len={len(ep_actions)}")

            obs_dict, _ = env.reset()
            obs = extract_policy_obs(obs_dict).to(device)
            ep_obs, ep_states, ep_actions, ep_success = _reset_ep_buffers()
            first_success_step = None
        else:
            obs = next_obs

    n = len(collected)
    print(f"\n[INFO] Collected {n}/{ep_count} episodes succeeded.")

    if n == 0:
        print("[WARN] No demos collected. Not saving.")
        env.close()
        return

    # --- HDF5 + JSON (RFCL ReplayDataset 형식) ---
    h5_path   = os.path.join(out_dir, "demos.h5")
    json_path = os.path.join(out_dir, "demos.json")
    pkl_path  = os.path.join(out_dir, "states_dataset.pkl")

    with h5py.File(h5_path, "w") as f:
        for i, demo in enumerate(collected):
            g = f.create_group(f"traj_{i}")
            g.create_dataset("obs",     data=demo["obs"])
            g.create_dataset("actions", data=demo["actions"])
            g.create_dataset("success", data=demo["success"])
            g.create_dataset("rewards", data=demo["rewards"])
            g.create_dataset("env_states", data=demo["states"])

    meta = {"episodes": [
        {"episode_id": i, "success": True, "reset_kwargs": {}, "episode_seed": None}
        for i in range(n)
    ]}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- states_dataset.pkl (InitialStateWrapper 형식) ---
    states_dataset = {
        i: {"state": demo["states"], "seed": None, "reset_kwargs": {}}
        for i, demo in enumerate(collected)
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(states_dataset, f)

    print(f"[INFO] Saved → {h5_path}")
    print(f"[INFO]         {json_path}")
    print(f"[INFO]         {pkl_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
