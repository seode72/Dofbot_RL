"""
load_best_agent.py
==================
best_agent.pt 체크포인트에서 PolicyModel 및 CriticModel 가중치를 로드하는 스크립트.

사용법:
    python load_best_agent.py
    python load_best_agent.py --checkpoint path/to/best_agent.pt --device cuda
"""

import argparse
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 모델 구조 정의 (train.py / models/ 와 동일)
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, output_dim: int, hidden_dims=(256, 128, 64), activation="elu") -> nn.Sequential:
    act_map = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}
    act_cls = act_map[activation.lower()]
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """PolicyModel과 동일한 구조 (skrl 없이 순수 PyTorch)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 128, 64), log_std_init: float = -1.0, device="cpu"):
        super().__init__()
        self.net = build_mlp(obs_dim, action_dim, hidden_dims)
        self.log_std_parameter = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.to(device)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean_actions : tanh로 클리핑된 행동 평균 (B, action_dim)
            log_std      : 로그 표준편차 (action_dim,)
        """
        mean_actions = torch.tanh(self.net(obs))
        return mean_actions, self.log_std_parameter


class CriticNetwork(nn.Module):
    """CriticModel과 동일한 구조 (obs + action → Q-value)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 128, 64), device="cpu"):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, 1, hidden_dims)
        self.to(device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# 로드 함수
# ---------------------------------------------------------------------------

def load_best_agent(
    checkpoint_path: str,
    obs_dim: int = 17,      # joint_pos(7) + joint_vel(7) + cube_pos(3)
    action_dim: int = 7,
    num_critics: int = 10,
    hidden_dims: tuple = (256, 128, 64),
    device: str = "cpu",
):
    """
    best_agent.pt 로부터 PolicyNetwork 및 CriticNetwork 목록을 로드합니다.

    Returns
    -------
    policy        : PolicyNetwork (eval 모드)
    critics       : list[CriticNetwork]  (10개, eval 모드)
    target_critics: list[CriticNetwork]  (10개, eval 모드)
    meta          : dict  (step, completed_eps, current_stage, best_succ_rate)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"[INFO] Checkpoint keys : {list(checkpoint.keys())}")
    print(f"[INFO] step            : {checkpoint.get('step')}")
    print(f"[INFO] completed_eps   : {checkpoint.get('completed_eps')}")
    print(f"[INFO] current_stage   : {checkpoint.get('current_stage')}")
    print(f"[INFO] best_succ_rate  : {checkpoint.get('best_succ_rate'):.2f}%")

    # --- Policy ---
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dims, device=device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    print("[INFO] Policy weights loaded.")

    # --- Critics & Target Critics ---
    critics = []
    target_critics = []
    for i in range(1, num_critics + 1):
        critic = CriticNetwork(obs_dim, action_dim, hidden_dims, device=device)
        critic.load_state_dict(checkpoint[f"critic_{i}"])
        critic.eval()
        critics.append(critic)

        target = CriticNetwork(obs_dim, action_dim, hidden_dims, device=device)
        target.load_state_dict(checkpoint[f"target_critic_{i}"])
        target.eval()
        target_critics.append(target)

    print(f"[INFO] {num_critics} critics + {num_critics} target critics loaded.")

    meta = {
        "step": checkpoint.get("step"),
        "completed_eps": checkpoint.get("completed_eps"),
        "current_stage": checkpoint.get("current_stage"),
        "best_succ_rate": checkpoint.get("best_succ_rate"),
    }

    return policy, critics, target_critics, meta


# ---------------------------------------------------------------------------
# 간단한 추론 예시
# ---------------------------------------------------------------------------

def demo_inference(policy: PolicyNetwork, obs_dim: int = 17, device: str = "cpu"):
    """랜덤 관측값으로 정책 추론을 테스트합니다."""
    dummy_obs = torch.randn(1, obs_dim, device=device)
    with torch.no_grad():
        mean_action, log_std = policy(dummy_obs)
    print(f"[DEMO] obs shape    : {dummy_obs.shape}")
    print(f"[DEMO] action shape : {mean_action.shape}  values: {mean_action.cpu().numpy()}")
    print(f"[DEMO] log_std      : {log_std.detach().cpu().numpy()}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = (
    "Dofbot_RL-HSH/Dofbot_RL-HSH/20260415_003530/best_agent.pt"
)


def main():
    parser = argparse.ArgumentParser(description="Load best_agent.pt weights")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="best_agent.pt 파일 경로")
    parser.add_argument("--device", type=str, default="cpu",
                        help="로드할 디바이스 (cpu / cuda)")
    parser.add_argument("--obs_dim", type=int, default=17)
    parser.add_argument("--action_dim", type=int, default=7)
    args = parser.parse_args()

    policy, critics, target_critics, meta = load_best_agent(
        checkpoint_path=args.checkpoint,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        device=args.device,
    )

    demo_inference(policy, obs_dim=args.obs_dim, device=args.device)
    print("\n[DONE] 모델 로드 완료.")
    print(f"       best_succ_rate = {meta['best_succ_rate']:.2f}%  |  step = {meta['step']}")


if __name__ == "__main__":
    main()
