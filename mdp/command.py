from isaaclab.managers import CommandTermCfg, CommandTerm
from isaaclab.utils import configclass
import torch


class TargetCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]
        self.target = env.scene[cfg.target_name]

        # 내부 command tensor
        self._command = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

        # metrics dictionary에 하나 등록
        self.metrics["target_dist"] = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: torch.Tensor):
        target_pos = self.target.data.root_pos_w[env_ids]

        self._command[env_ids, 0] = target_pos[:, 0]
        self._command[env_ids, 1] = target_pos[:, 1]
        self._command[env_ids, 2] = 0.0

    def _update_command(self):
        # resampling_time_range를 크게 줬다면
        # episode 중간에 굳이 갱신 안 해도 됨
        pass

    def _update_metrics(self):
        robot_pos = self.robot.data.root_pos_w[:, :2]
        target_xy = self._command[:, :2]
        self.metrics["target_dist"] = torch.norm(target_xy - robot_pos, dim=-1)


@configclass
class TargetCommandCfg(CommandTermCfg):
    class_type: type = TargetCommand

    asset_name: str = "robot"
    target_name: str = "target_plane"
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)