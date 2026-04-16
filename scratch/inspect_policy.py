
import torch
import gymnasium as gym
from models.policy import PolicyModel
from models.models_cfg import PolicyModelCfg

def inspect_policy():
    obs_dim = 17    # 35
    action_dim = 7
    device = "cpu"
    
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_dim,))
    action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))
    
    cfg = PolicyModelCfg()
    policy = PolicyModel(observation_space, action_space, cfg, device)
    
    states = torch.randn(1, obs_dim)
    
    print("Testing policy.act()...")
    outputs = policy.act({"states": states}, role="policy")
    print(f"act() returned {len(outputs)} values: {type(outputs)}")
    for i, out in enumerate(outputs):
        if isinstance(out, torch.Tensor):
            print(f"  [{i}] Tensor shape: {out.shape}")
        else:
            print(f"  [{i}] Type: {type(out)}")

if __name__ == "__main__":
    inspect_policy()
