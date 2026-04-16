import gymnasium as gym

print("[DOFBOT] dofbot package imported")
print("[DOFBOT] registering Isaac-Dofbot-v0")

gym.register(
    id="Isaac-Dofbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "dofbot_task.dofbot_env_cfg:DofbotEnvCfg",
    },
)