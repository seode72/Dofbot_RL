import torch
from skrl.models.torch import Model, DeterministicMixin

from .mlp import build_mlp
from .models_cfg import CriticModelCfg

class CriticModel(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        cfg: CriticModelCfg,
        device: str,
        clip_actions: bool = False,
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = build_mlp(
            input_dim=self.num_observations + self.num_actions,
            output_dim=1,
            hidden_dims=cfg.network.hidden_dims,
            activation=cfg.network.activation,
        ).to(self.device)

    def compute(self, inputs, role=""):
        states = inputs["states"]
        actions = inputs.get("taken_actions", None)
        
        if actions is None:
            # skrl fallback in some cases: when actions is not provided, 
            # it might happen during initialization checks. We handle it safely.
            actions = torch.zeros((states.shape[0], self.num_actions), device=self.device)

        x = torch.cat([states, actions], dim=-1)
        q_value = self.net(x)
        return q_value, {}
