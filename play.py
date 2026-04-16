from __future__ import annotations

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# argparse
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play DOFBOT with trained SAC.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Dofbot-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to load.")
parser.add_argument("--cube_y", type=float, default=None,
                    help="Cube local Y position override (m). None = use env_cfg default. "
                         "0.10 = curriculum start, 0.20 = original target.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
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

import dofbot_task
from dofbot_task.dofbot_env_cfg import DofbotEnvCfg
from checkpoint_tools.checkpoint import load_checkpoint

from models.policy import PolicyModel
from models.critic import CriticModel
from models.models_cfg import PolicyModelCfg, CriticModelCfg


def extract_policy_obs(obs_dict) -> torch.Tensor:
    """train.py와 동일한 17-dim obs 구성. 반드시 일치해야 checkpoint shape가 맞음."""
    policy_obs = obs_dict["policy"]
    obs_parts = [
        policy_obs["joint_pos"],             # [N, 7]
        policy_obs["joint_vel"],             # [N, 7]
        policy_obs["cube_pos"],              # [N, 3]
    ]
    return torch.cat(obs_parts, dim=-1)  # [N, 17]


def set_cube_y(base_env, cube_y: float, device: str) -> None:
    """모든 환경의 cube를 지정한 local Y 위치로 재배치."""
    cube = base_env.scene["cube"]
    origins = base_env.scene.env_origins  # [num_envs, 3]
    num_envs = origins.shape[0]

    state = torch.zeros(num_envs, 13, device=device)
    state[:, 0] = origins[:, 0] + 0.0
    state[:, 1] = origins[:, 1] + cube_y
    state[:, 2] = origins[:, 2] + 0.03
    state[:, 3] = 1.0  # qw (identity)

    cube.write_root_state_to_sim(state)


def main():

    runtime_device = args_cli.device
    if str(runtime_device).startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device '{args_cli.device}' but CUDA is unavailable. Falling back to CPU.")
        runtime_device = "cpu"

    env_cfg = DofbotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = runtime_device

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    # obs_dim / action_dim을 먼저 파악하기 위해 초기 reset
    obs_dict, _ = env.reset()
    obs_dim = extract_policy_obs(obs_dict).shape[-1]
    action_dim = base_env.action_manager.total_action_dim

    policy_cfg = PolicyModelCfg()
    critic_cfg = CriticModelCfg()

    policy = PolicyModel(
        observation_space=obs_dim,
        action_space=action_dim,
        cfg=policy_cfg,
        device=runtime_device,   
    )

    # train.py와 동일하게 NUM_QS=10개의 critic/target_critic 생성 (체크포인트 형식 일치)
    NUM_QS = 10
    critics = [CriticModel(obs_dim, action_dim, critic_cfg, runtime_device) for _ in range(NUM_QS)]
    target_critics = [CriticModel(obs_dim, action_dim, critic_cfg, runtime_device) for _ in range(NUM_QS)]

    models = {"policy": policy}
    for i, c in enumerate(critics):
        models[f"critic_{i+1}"] = c
    for i, tc in enumerate(target_critics):
        models[f"target_critic_{i+1}"] = tc

    from dofbot_task.agent.sac import SAC, SAC_DEFAULT_CONFIG, RandomMemory

    sac_cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    memory = RandomMemory(
        memory_size=1000,
        num_envs=args_cli.num_envs,
        device=runtime_device,  
    )

    agent = SAC(
        models=models,
        memory=memory,
        observation_space=obs_dim,
        action_space=action_dim,
        device=runtime_device,   
        cfg=sac_cfg,
    )
    agent.init()

    # 안전하고 완전한 checkpoint 복원
    load_checkpoint(agent, args_cli.checkpoint, runtime_device)
    print(f"[INFO] Loaded SAC checkpoint from: {args_cli.checkpoint}")

    obs_dict, _ = env.reset()
    # cube_y override: 평가 거리 지정
    if args_cli.cube_y is not None:
        set_cube_y(base_env, args_cli.cube_y, runtime_device)
        base_env.scene.write_data_to_sim()  # cube 상태 동기화
        obs_dict = base_env.observation_manager.compute()  # 변경된 상태 기반 obs 재계산
        print(f"[INFO] Cube overridden to local Y={args_cli.cube_y:.2f}m")
    obs = extract_policy_obs(obs_dict).to(runtime_device)

    # random_timesteps 조건을 우회하기 위해 timestep을 충분히 크게 설정
    PLAY_TIMESTEP = agent._random_timesteps + 1

    while simulation_app.is_running():
        with torch.no_grad():
            actions, _, _ = agent.act(
                states=obs,
                timestep=PLAY_TIMESTEP,
                timesteps=PLAY_TIMESTEP,
            )

        next_obs_dict, rewards, terminated, truncated, _ = env.step(actions)
        next_obs = extract_policy_obs(next_obs_dict).to(runtime_device)

        dones = terminated | truncated
        if dones.all():
            obs_dict, _ = env.reset()
            if args_cli.cube_y is not None:
                set_cube_y(base_env, args_cli.cube_y, runtime_device)
                base_env.scene.write_data_to_sim()
                obs_dict = base_env.observation_manager.compute()
            obs = extract_policy_obs(obs_dict).to(runtime_device)
        else:
            obs = next_obs

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()