"""
Standalone Soft Actor-Critic (SAC) Agent
========================================
skrl Agent 상속 없이 독립적으로 동작하는 SAC 구현.
skrl Model/Memory 인터페이스만 사용하며, base class 호환성 문제를 완전히 제거.
학습 로직은 원본 커스텀 SAC와 100% 동일.
"""
from typing import Any, Mapping, Optional, Tuple, Union

import copy
from dofbot_task.agent.replay_dataset import ReplayDataset, combine
import itertools
import gymnasium
from packaging import version
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.memories.torch import Memory
from skrl.models.torch import Model


SAC_DEFAULT_CONFIG = {
    "gradient_steps": 1,          # train.py에서 80으로 오버라이드
    "batch_size": 256,
    "num_qs": 10,                  # 원본 rfcl: Q 앙상블 개수
    "num_min_qs": 2,               # 원본 rfcl: target 계산 시 최솟값 사용 개수
    "actor_update_freq": 20,       # 원본 rfcl: critic 20번당 actor 1번

    "discount_factor": 0.99,
    "polyak": 0.005,

    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "learning_rate_scheduler": None,
    "learning_rate_scheduler_kwargs": {},

    "state_preprocessor": None,
    "state_preprocessor_kwargs": {},

    "random_timesteps": 1000,
    "learning_starts": 5000,    # 5000 step 이후 학습 (Off-Policy)

    "grad_norm_clip": 1.0,

    "learn_entropy": True,
    "entropy_learning_rate": 3e-4,
    "initial_entropy_value": 1.0,          # 원본 rfcl: initial_temperature=1.0
    "target_entropy": None,

    "rewards_shaper": None,
    "mixed_precision": False,

    "experiment": {
        "directory": "logs",
        "experiment_name": "sac_dofbot",
        "write_interval": 1000,
        "checkpoint_interval": 5000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }
}


class SAC:
    """Standalone Soft Actor-Critic (SAC) — skrl Agent base class 비상속."""

    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Memory] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        self.cfg = _cfg

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # spaces
        self.observation_space = observation_space
        self.action_space = action_space

        # models
        self.models = models
        self.policy = models.get("policy", None)

        # 앙상블 critic: critic_1 ~ critic_N 동적 탐지 (원본 rfcl: num_qs=10)
        self.critics: list = []
        self.target_critics: list = []
        i = 1
        while f"critic_{i}" in models:
            self.critics.append(models[f"critic_{i}"])
            i += 1
        i = 1
        while f"target_critic_{i}" in models:
            self.target_critics.append(models[f"target_critic_{i}"])
            i += 1

        # 하위 호환 aliases
        self.critic_1 = self.critics[0] if len(self.critics) > 0 else None
        self.critic_2 = self.critics[1] if len(self.critics) > 1 else None
        self.target_critic_1 = self.target_critics[0] if len(self.target_critics) > 0 else None
        self.target_critic_2 = self.target_critics[1] if len(self.target_critics) > 1 else None

        # memory
        self.memory = memory
        self.offline_buffer: ReplayDataset | None = None

        # checkpoint modules (dynamic)
        self.checkpoint_modules: dict = {"policy": self.policy}
        for i, c in enumerate(self.critics):
            self.checkpoint_modules[f"critic_{i+1}"] = c
        for i, tc in enumerate(self.target_critics):
            self.checkpoint_modules[f"target_critic_{i+1}"] = tc

        # freeze and sync all target networks
        for critic, target in zip(self.critics, self.target_critics):
            target.freeze_parameters(True)
            target.update_parameters(critic, polyak=1)

        # configuration
        self._gradient_steps = _cfg["gradient_steps"]
        self._batch_size = _cfg["batch_size"]
        self._num_qs = _cfg.get("num_qs", len(self.critics))
        self._num_min_qs = _cfg.get("num_min_qs", 2)
        self._actor_update_freq = _cfg.get("actor_update_freq", 1)
        self._grad_step_count = 0
        self._discount_factor = _cfg["discount_factor"]
        self._polyak = _cfg["polyak"]
        self._actor_learning_rate = _cfg["actor_learning_rate"]
        self._critic_learning_rate = _cfg["critic_learning_rate"]
        self._learning_rate_scheduler = _cfg["learning_rate_scheduler"]
        self._state_preprocessor = _cfg["state_preprocessor"]
        self._random_timesteps = _cfg["random_timesteps"]
        self._learning_starts = _cfg["learning_starts"]
        self._grad_norm_clip = _cfg["grad_norm_clip"]
        self._entropy_learning_rate = _cfg["entropy_learning_rate"]
        self._learn_entropy = _cfg["learn_entropy"]
        self._entropy_coefficient = _cfg["initial_entropy_value"]
        self._rewards_shaper = _cfg["rewards_shaper"]
        self._mixed_precision = _cfg["mixed_precision"]

        # device type for autocast
        self._device_type = self.device.type

        # mixed precision scaler
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = _cfg["target_entropy"]
            if self._target_entropy is None:
                # 원본 rfcl: target_entropy = -act_dims / 2
                if isinstance(action_space, int):
                    self._target_entropy = -float(action_space) / 2
                elif isinstance(action_space, (tuple, list)):
                    self._target_entropy = -float(np.prod(action_space)) / 2
                elif issubclass(type(action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(action_space.shape).astype(np.float32) / 2
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coefficient], lr=self._entropy_learning_rate)
            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # optimizers — critic optimizer는 앙상블 전체 파라미터를 합침
        if self.policy is not None and len(self.critics) > 0:
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate)
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(*[c.parameters() for c in self.critics]),
                lr=self._critic_learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **_cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **_cfg["learning_rate_scheduler_kwargs"]
                )
            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # preprocessor
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**_cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = lambda x, **kw: x  # identity

        # tracking (간단한 write_interval 지원)
        self.write_interval = _cfg["experiment"].get("write_interval", 1000)
        self._tracking_data = {}

    def init(self, trainer_cfg=None) -> None:
        """메모리 텐서 생성."""
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            # offline_buffer 는 ReplayDataset 이므로 별도 tensor 생성 불필요

            self._tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

    def _process_model_output(self, output, role=""):
        """모델의 act() 반환값 개수에 상관없이 (actions, log_prob, outputs)를 추출."""
        if isinstance(output, tuple):
            if len(output) == 3:
                return output  # Standard: (actions, log_prob, outputs)
            elif len(output) == 2:
                # Some skrl versions/mixins return (actions, outputs)
                actions, outputs = output
                log_prob = None
                if isinstance(outputs, dict):
                    log_prob = outputs.get("log_prob", None)
                return actions, log_prob, outputs
        
        # Fallback for single return or unexpected format
        return output, None, {}

    def act(self, states: torch.Tensor, timestep: int, timesteps: int):
        """환경 상태로부터 행동 결정."""
        # 디바이스 통일 보장
        if states.device != self.device:
            states = states.to(self.device)
        processed_states = self._state_preprocessor(states)

        # 랜덤 탐색
        if timestep < self._random_timesteps:
            batch_size = processed_states.shape[0]
            actions = 2.0 * torch.rand(
                batch_size,
                self.policy.num_actions,
                device=self.device,
                dtype=torch.float32,
            ) - 1.0
            return actions, None, {}

        # 정책 기반 행동
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            output = self.policy.act({"states": processed_states}, role="policy")
            actions, _, outputs = self._process_model_output(output, role="policy")
            
        return actions, None, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """경험 리플레이 버퍼에 전이 저장."""
        if self.memory is not None:
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """학습 스텝 트리거."""
        if timestep >= self._learning_starts:
            self._set_mode("train")
            self._update(timestep, timesteps)
            self._set_mode("eval")

    def update(self, timestep: int, timesteps: int) -> None:
        return self._update(timestep, timesteps)

    def _set_mode(self, mode: str) -> None:
        """모델 train/eval 모드 전환 (skrl 버전 호환성 고려)."""
        for name, model in self.models.items():
            if model is not None:
                try:
                    if mode == "train":
                        model.train()
                        if hasattr(model, "set_running_mode"):
                            model.set_running_mode("train")
                    else:
                        model.eval()
                        if hasattr(model, "set_running_mode"):
                            model.set_running_mode("eval")
                except Exception as e:
                    # 일부 skrl 모델은 특정 모드에서 예외가 발생할 수 있으므로 안전하게 처리
                    pass

    def track_data(self, tag: str, value: float) -> None:
        self._tracking_data[tag] = value

    def _get_q(self, model, states, actions, role: str) -> torch.Tensor:
        """critic 모델에서 Q값만 추출."""
        out = model.act({"states": states, "taken_actions": actions}, role=role)
        return out[0] if isinstance(out, (tuple, list)) else out

    def _update(self, timestep: int, timesteps: int) -> None:
        """SAC 메인 업데이트 — 원본 rfcl 로직 기반.

        원본 rfcl 주요 설정:
          - grad_updates_per_step=80: env step 1번당 gradient update 80회
          - actor_update_freq=20: critic 20번당 actor 1번 업데이트
          - num_qs=10, num_min_qs=2: Q 앙상블 10개, target은 랜덤 2개 min
        """
        last_critic_loss = None
        last_policy_loss = None
        last_entropy_loss = None
        last_target_values = None
        last_q_mean = None

        # 한 번 호출에 여러 Gradient Step 가능
        for gradient_step in range(self._gradient_steps):
            # ---- 샘플링: 원본 rfcl 50:50 인터리빙 ----
            # Offline Buffer가 있으면, Hybrid 학습 수행
            online_size = len(self.memory)
            if self.offline_buffer is not None and len(self.offline_buffer) > 0:
                if online_size == 0:
                    # Stage 2 전환 직후: online buffer가 비어있으면 offline만 사용
                    batch = self.offline_buffer.sample_random_batch(self._batch_size)
                else:
                    half = self._batch_size // 2    # 50%
                    online_raw = self.memory.sample(names=self._tensors_names, batch_size=half)[0]
                    on_dones = online_raw[4].view(-1) | online_raw[5].view(-1)
                    on_masks = ((~on_dones) | online_raw[5].view(-1)).float()
                    online_batch = {
                        "states":      online_raw[0].view(half, -1),
                        "actions":     online_raw[1].view(half, -1),
                        "rewards":     online_raw[2].view(half),
                        "next_states": online_raw[3].view(half, -1),
                        "masks":       on_masks,
                    }
                    offline_batch = self.offline_buffer.sample_random_batch(half)
                    batch = combine(online_batch, offline_batch)
            else:
                online_raw = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]
                on_dones = online_raw[4].view(-1) | online_raw[5].view(-1)
                on_masks = ((~on_dones) | online_raw[5].view(-1)).float()
                batch = {
                    "states":      online_raw[0].view(self._batch_size, -1),
                    "actions":     online_raw[1].view(self._batch_size, -1),
                    "rewards":     online_raw[2].view(self._batch_size),
                    "next_states": online_raw[3].view(self._batch_size, -1),
                    "masks":       on_masks,
                }

            sampled_states      = self._state_preprocessor(batch["states"])
            sampled_next_states = self._state_preprocessor(batch["next_states"])
            sampled_actions     = batch["actions"]
            sampled_rewards     = batch["rewards"].unsqueeze(-1)
            sampled_masks       = batch["masks"].unsqueeze(-1)

            # ---- Critic update (매 gradient step) ----
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                with torch.no_grad():
                    next_actions, _, _ = self._process_model_output(
                        self.policy.act({"states": sampled_next_states}, role="policy")
                    )
                    # 원본 rfcl: num_min_qs개의 target critic을 랜덤 샘플 후 min
                    target_indices = np.random.choice(
                        len(self.target_critics), size=self._num_min_qs, replace=False
                    )
                    target_qs = torch.stack([
                        self._get_q(self.target_critics[idx], sampled_next_states, next_actions,
                                    f"target_critic_{idx+1}")
                        for idx in target_indices
                    ], dim=0)
                    next_q = torch.min(target_qs, dim=0).values
                    target_values = sampled_rewards + self._discount_factor * sampled_masks * next_q
                    """actor loss에는 entropy 사용, critic backup에는 entropy 미사용"""

                # 원본 rfcl: 모든 critic의 MSE 평균
                critic_loss = sum(
                    F.mse_loss(
                        self._get_q(c, sampled_states, sampled_actions, f"critic_{i+1}"),
                        target_values
                    )
                    for i, c in enumerate(self.critics)
                ) / len(self.critics)

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(*[c.parameters() for c in self.critics]),
                    self._grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)
            last_critic_loss = critic_loss
            last_target_values = target_values

            # 모든 target critic polyak update
            for critic, target in zip(self.critics, self.target_critics):
                target.update_parameters(critic, polyak=self._polyak)

            # ---- Actor + Entropy update (actor_update_freq마다) ----
            if self._grad_step_count % self._actor_update_freq == 0:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    actions, log_prob, _ = self._process_model_output(
                        self.policy.act({"states": sampled_states}, role="policy")
                    )
                    # 원본 rfcl: actor loss에서 앙상블 Q의 mean 사용
                    q_mean = torch.mean(torch.stack([
                        self._get_q(c, sampled_states, actions, f"critic_{i+1}")
                        for i, c in enumerate(self.critics)
                    ], dim=0), dim=0)
                    policy_loss = (self._entropy_coefficient * log_prob - q_mean).mean()

                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                self.scaler.step(self.policy_optimizer)
                last_policy_loss = policy_loss
                last_q_mean = q_mean

                if self._learn_entropy:
                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        entropy_loss = -(
                            self.log_entropy_coefficient * (log_prob.detach() + self._target_entropy)
                        ).mean()
                    self.entropy_optimizer.zero_grad()
                    self.scaler.scale(entropy_loss).backward()
                    self.scaler.step(self.entropy_optimizer)
                    self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())
                    last_entropy_loss = entropy_loss

            self.scaler.update()

            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            self._grad_step_count += 1

        # ---- Tracking (마지막 gradient step 기준) ----
        if self.write_interval > 0:
            if last_critic_loss is not None:
                self.track_data("Loss / Critic loss", last_critic_loss.item())
            if last_policy_loss is not None:
                self.track_data("Loss / Policy loss", last_policy_loss.item())
                self.track_data("Q-network / Q mean (all critics)", torch.mean(last_q_mean).item())
            if last_target_values is not None:
                self.track_data("Target / Target (mean)", torch.mean(last_target_values).item())
            if last_entropy_loss is not None:
                self.track_data("Loss / Entropy loss", last_entropy_loss.item())
                self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

    # ------------------------------------------------------------------ #
    # Checkpoint save / load
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, extra: dict = None) -> None:
        """체크포인트 저장."""
        state = {}
        for name, module in self.checkpoint_modules.items():
            if module is not None and hasattr(module, "state_dict"):
                state[name] = module.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        """체크포인트 로드. 추가 메타데이터를 반환."""
        ckpt = torch.load(path, map_location=self.device)
        for name, module in self.checkpoint_modules.items():
            if name in ckpt and module is not None and hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f"[WARN] Failed to load '{name}': {e}")
        return ckpt

# 랜덤 샘플링 Replay Buffer
class RandomMemory(Memory):
    """Random sampling replay buffer."""
    def __init__(
        self,
        *,
        memory_size: int,
        num_envs: int = 1,
        device: str | torch.device | None = None,
        export: bool = False,
        export_format: Literal["pt", "npz", "csv"] = "pt",
        export_directory: str = "",
        replacement: bool = True,
    ) -> None:
        super().__init__(
            memory_size=memory_size,
            num_envs=num_envs,
            device=device,
            export=export,
            export_format=export_format,
            export_directory=export_directory,
        )
        self._replacement = replacement

    def sample(
        self, names: list[str], *, batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> list[list[torch.Tensor]]:
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()
        if self._replacement:
            indexes = torch.randint(0, size, (batch_size,))
        else:
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]
        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)
        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)