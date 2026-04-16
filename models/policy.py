import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin

from .mlp import build_mlp
from .models_cfg import PolicyModelCfg


class PolicyModel(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg: PolicyModelCfg,
        device: str,
        clip_actions: bool = True,
        clip_log_std: bool = True,
        min_log_std: float = -5.0,
        max_log_std: float = -1.0,
        reduction: str = "sum",
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = build_mlp(
            input_dim=self.num_observations,
            output_dim=self.num_actions,
            hidden_dims=cfg.network.hidden_dims,
            activation=cfg.network.activation,
        ).to(self.device)

        self.log_std_parameter = nn.Parameter(
            torch.full(
                size=(self.num_actions,),
                fill_value=cfg.log_std_init,
                device=self.device,
            ),
            requires_grad=True,
        )

        # GaussianMixin action limits (required for clip_actions=True)
        self._g_min_actions = torch.full(size=(self.num_actions,), fill_value=-1.0, device=self.device)
        self._g_max_actions = torch.full(size=(self.num_actions,), fill_value=1.0, device=self.device)

        
    def compute(self, inputs, role=""):
        states = inputs["states"]
        mean_actions = torch.tanh(self.net(states))
        return mean_actions, {"log_std": self.log_std_parameter}