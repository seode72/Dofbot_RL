import torch
from skrl.models.torch import Model, DeterministicMixin

from .mlp import build_mlp
from .models_cfg import ValueModelCfg

class ValueModel(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg: ValueModelCfg,
        device: str,
        clip_actions: bool = False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = build_mlp(
            input_dim=self.num_observations,
            output_dim=1,
            hidden_dims=cfg.network.hidden_dims,
            activation=cfg.network.activation,
        )

    def compute(self, inputs, role=""):
        states = inputs["states"]
        value = self.net(states)
        return value, {}