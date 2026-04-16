"""
replay_demos.py
===============
collect_demos.py로 저장한 demos를 IsaacLab 환경에서 재실행하여
dual_finger_contact 기반 성공 여부를 검증한다.

각 데모의 초기 상태(states_dataset.pkl의 첫 번째 state)로 환경을 리셋한 뒤,
저장된 action sequence를 그대로 재생하고 성공 여부를 확인한다.

성공 기준: 재생 도중 reward_dual_finger_contact > 0 인 스텝이 하나라도 있으면 성공.

사용법:
    ./isaaclab.sh -p scripts/dofbot_0331/replay_demos.py \\
        --demos_dir demos --task Isaac-Dofbot-v0
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task",      type=str, default="Isaac-Dofbot-v0")
parser.add_argument("--demos_dir", type=str, default="demos",
                    help="collect_demos.py가 저장한 디렉토리")
parser.add_argument("--seed",      type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after launch ---
import os
import pickle

import h5py
import numpy as np
import torch
import gymnasium as gym

import dofbot_task
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from mdp.reward import reward_dual_finger_contact

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ARM_JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "Wrist_Twist_RevoluteJoint",
    "Finger_Left_01_RevoluteJoint",
    "Finger_Right_01_RevoluteJoint",
]


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


def restore_env_state(base_env, state_np: np.ndarray, joint_ids: list[int], device: str):
    """
    70-dim state 배열로 환경을 복원한다.

    Layout:
      [0:7]   joint_pos
      [7:14]  joint_vel
      [14:17] cube_local_pos
      [17:21] cube_quat
      [21:24] cube_lin_vel
      [24:27] cube_ang_vel
      [27:45] left_contact_hist  [18] - 저장만 하고 직접 주입 불가 (Physics 읽기 전용)
      [45:63] right_contact_hist [18] - 동일
      [63:70] prev_action        [7]

    Note:
      contact history는 PhysX에서 매 스텝 덮어쓰므로 직접 주입할 수 없다.
      env.reset() 이후 호출하면 contact history는 0으로 초기화된 상태가 된다.
      step 0부터 재생하는 경우엔 이것으로 충분하다.
      임의 t_i에서 시작하는 RFCL 커리큘럼을 쓰려면 이전 6 action을 warm-up으로
      재시뮬레이션하는 별도 로직이 필요하다 (TODO).
    """
    robot  = base_env.scene["robot"]
    cube   = base_env.scene["cube"]
    origin = base_env.scene.env_origins[0]

    # 1. 로봇 관절 복원
    joint_pos = torch.tensor(state_np[0:7],  dtype=torch.float32, device=device).unsqueeze(0)
    joint_vel = torch.tensor(state_np[7:14], dtype=torch.float32, device=device).unsqueeze(0)
    env_ids   = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)

    # 2. Cube 상태 복원
    cube_local_pos = torch.tensor(state_np[14:17], dtype=torch.float32, device=device)
    cube_quat      = torch.tensor(state_np[17:21], dtype=torch.float32, device=device)
    cube_lin_vel   = torch.tensor(state_np[21:24], dtype=torch.float32, device=device)
    cube_ang_vel   = torch.tensor(state_np[24:27], dtype=torch.float32, device=device)
    cube_state     = torch.zeros(1, 13, device=device)
    cube_state[0, 0:3]   = cube_local_pos + origin
    cube_state[0, 3:7]   = cube_quat
    cube_state[0, 7:10]  = cube_lin_vel
    cube_state[0, 10:13] = cube_ang_vel
    cube.write_root_state_to_sim(cube_state)

    # 3. prev_action 복원 (action_rate_l2 penalty 정확도)
    prev_action = torch.tensor(state_np[63:70], dtype=torch.float32, device=device).unsqueeze(0)
    try:
        base_env.action_manager._prev_action[:] = prev_action
    except Exception:
        pass  # 속성명이 버전에 따라 다를 수 있음, 실패해도 치명적이지 않음

    # 4. PhysX 동기화 (contact history는 다음 스텝에서 자동으로 덮어써짐)
    base_env.scene.write_data_to_sim()


def main():
    device = args_cli.device

    demos_dir = os.path.join(SCRIPT_DIR, args_cli.demos_dir)
    h5_path   = os.path.join(demos_dir, "demos.h5")
    pkl_path  = os.path.join(demos_dir, "states_dataset.pkl")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"demos.h5 not found: {h5_path}\n"
                                f"collect_demos.py를 먼저 실행하세요.")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"states_dataset.pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        states_dataset = pickle.load(f)

    print(f"[INFO] Loaded {len(states_dataset)} demos from {demos_dir}")

    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = device

    env      = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset()

    joint_ids = get_joint_ids(base_env)

    # --- 각 데모 재생 ---
    results = []
    with h5py.File(h5_path, "r") as f:
        demo_ids = sorted(states_dataset.keys())
        for demo_id in demo_ids:
            if not simulation_app.is_running():
                break

            grp     = f[f"traj_{demo_id}"]
            actions = np.array(grp["actions"])   # [T, 7]
            states  = states_dataset[demo_id]["state"]  # [T+1, 27]

            T = len(actions)

            # 초기 상태로 환경 리셋
            env.reset()
            restore_env_state(base_env, states[0], joint_ids, device)
            base_env.observation_manager.compute()  # 센서 상태 동기화

            # 저장된 action 재생
            success_steps = 0
            for t in range(T):
                if not simulation_app.is_running():
                    break

                action_t = torch.tensor(actions[t], dtype=torch.float32, device=device).unsqueeze(0)
                env.step(action_t)

                dual = reward_dual_finger_contact(base_env)[0].item()
                if dual > 0.0:
                    success_steps += 1

            success = success_steps > 0
            results.append(success)
            print(
                f"[Demo {demo_id:>3}] {'SUCCESS' if success else 'FAIL   '}"
                f"  len={T}  success_steps={success_steps}"
            )

    n_total   = len(results)
    n_success = sum(results)
    print(f"\n[RESULT] {n_success}/{n_total} demos verified as successful.")
    print(f"[RESULT] Success rate: {n_success/n_total*100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
