"""
ReplayDataset — 원본 rfcl/rfcl/data/dataset.py 의 PyTorch 이식.
=================================================================
데모 데이터를 단순한 dict of Tensors 로 메모리에 보관하고,
sample_random_batch() 로 flat 인덱싱 기반 랜덤 샘플링을 수행.
skrl Memory 의존성 없이 독립적으로 동작.
"""

import h5py
import numpy as np
import torch


def combine(online_batch: dict, offline_batch: dict) -> dict:
    """
    원본 rfcl/rfcl/utils/tools.py 의 combine() 를 PyTorch로 이식.
    두 배치를 인터리빙(interleaving) 방식으로 병합.
    짝수 인덱스 = online, 홀수 인덱스 = offline.

    Args:
        online_batch: {key: Tensor(N, ...)}
        offline_batch: {key: Tensor(N, ...)}
    Returns:
        combined: {key: Tensor(2N, ...)}
    """
    combined = {}
    for k, v in online_batch.items():
        ov = offline_batch[k]
        total = v.shape[0] + ov.shape[0]
        tmp = torch.empty((total, *v.shape[1:]), dtype=v.dtype, device=v.device)
        tmp[0::2] = v
        tmp[1::2] = ov
        combined[k] = tmp
    return combined


class ReplayDataset:
    """
    H5 데모 파일로부터 (s, a, r, s', mask) 데이터를 로드하여
    flat tensor dict 로 보관하는 오프라인 버퍼.
    원본 rfcl 의 ReplayDataset 과 동일한 인터페이스.
    """

    def __init__(self, h5_path: str, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.data: dict[str, torch.Tensor] = {}
        self._size = 0
        self._load_from_h5(h5_path)

    def _load_from_h5(self, h5_path: str):
        all_states = []
        all_next_states = []
        all_actions = []
        all_rewards = []
        all_masks = []

        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                traj = f[key]
                obs = np.array(traj["obs"], dtype=np.float32)       # [T+1, obs_dim]
                actions = np.array(traj["actions"], dtype=np.float32)  # [T, act_dim]
                rewards = np.array(traj["rewards"], dtype=np.float32)  # [T]

                t_len = actions.shape[0]

                # NaN 정제 (Isaac Sim 센서 초기화 문제 대응)
                obs = np.nan_to_num(obs, nan=0.0)
                actions = np.nan_to_num(actions, nan=0.0)
                rewards = np.nan_to_num(rewards, nan=0.0)

                all_states.append(obs[:t_len])
                all_next_states.append(obs[1:t_len + 1])
                all_actions.append(actions)
                all_rewards.append(rewards)
                # 원본 rfcl: 데모 데이터는 전부 mask=1.0 (항상 bootstrap)
                all_masks.append(np.ones(t_len, dtype=np.float32))

        states = torch.tensor(np.concatenate(all_states), device=self.device)
        next_states = torch.tensor(np.concatenate(all_next_states), device=self.device)
        actions = torch.tensor(np.concatenate(all_actions), device=self.device)
        rewards = torch.tensor(np.concatenate(all_rewards), device=self.device)
        masks = torch.tensor(np.concatenate(all_masks), device=self.device)

        self.data = {
            "states": states,
            "next_states": next_states,
            "actions": actions,
            "rewards": rewards,
            "masks": masks,
        }
        self._size = states.shape[0]
        print(f"[ReplayDataset] Loaded {self._size} transitions from {h5_path}")

    def sample_random_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """랜덤 인덱스로 flat 샘플링 (원본과 동일)."""
        indices = torch.randint(0, self._size, (batch_size,))
        return {k: v[indices] for k, v in self.data.items()}

    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    # ---- Stage 2 전환용: Online 버퍼 데이터를 받아 자신을 교체 ----
    @classmethod
    def from_online_buffer(cls, online_memory, device: str | torch.device = "cpu"):
        """
        skrl RandomMemory 의 내용을 읽어와 ReplayDataset 으로 변환.
        원본 rfcl 의 `algo.offline_buffer = copy.deepcopy(algo.replay_buffer)` 에 해당.
        """
        obj = cls.__new__(cls)
        obj.device = torch.device(device)

        # skrl Memory 에서 유효한 데이터만 추출
        n = len(online_memory)
        if n == 0:
            obj.data = {}
            obj._size = 0
            return obj

        # sample_by_index 로 모든 유효 데이터를 가져옴
        all_idx = torch.arange(n)
        sampled = online_memory.sample_by_index(
            names=["states", "actions", "rewards", "next_states", "terminated", "truncated"],
            indexes=all_idx,
            mini_batches=1,
        )
        # sampled 는 list of list of tensors: [[states, actions, rewards, next_states, term, trunc]]
        tensors = sampled[0]

        states = tensors[0].to(device)
        actions = tensors[1].to(device)
        rewards = tensors[2].to(device)
        next_states = tensors[3].to(device)
        terminated = tensors[4].to(device)
        truncated = tensors[5].to(device)

        # mask 계산: 원본 rfcl 방식 — masks = ((~dones) | truncations).float()
        dones = (terminated.view(-1) | truncated.view(-1))
        masks = ((~dones) | truncated.view(-1)).float()

        obj.data = {
            "states": states.view(n, -1),
            "next_states": next_states.view(n, -1),
            "actions": actions.view(n, -1),
            "rewards": rewards.view(n),
            "masks": masks.view(n),
        }
        obj._size = n
        print(f"[ReplayDataset] Created from online buffer: {n} transitions")
        return obj

    # ---- 직렬화 (checkpoint 저장/로드) ----
    def state_dict(self) -> dict:
        return {"data": self.data, "size": self._size}

    def load_state_dict(self, state: dict):
        self.data = {k: v.to(self.device) for k, v in state["data"].items()}
        self._size = state["size"]
