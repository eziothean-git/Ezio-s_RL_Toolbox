import gymnasium as gym
from . import agents
from .env_cfg import G1NativeEnvCfg, G1NativeEnvCfg_PLAY

gym.register(
    id="myrl/Locomotion-Flat-G1Native-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1NativeEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1NativePPORunnerCfg",
    },
)
gym.register(
    id="myrl/Locomotion-Flat-G1Native-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1NativeEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1NativePPORunnerCfg",
    },
)
