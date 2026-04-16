from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class MLPConfig:
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"


@dataclass
class PolicyModelCfg:
    network: MLPConfig = field(default_factory=MLPConfig)
    log_std_init: float = -1.0



@dataclass
class ValueModelCfg:
    network: MLPConfig = field(default_factory=MLPConfig)

@dataclass
class CriticModelCfg:
    network: MLPConfig = field(default_factory=MLPConfig)