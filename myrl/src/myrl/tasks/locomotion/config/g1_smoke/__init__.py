import gymnasium as gym
from . import agents
from .env_cfg import G1SmokeEnvCfg, G1SmokeEnvCfg_PLAY

gym.register(
    id="myrl/Locomotion-Flat-G1Smoke-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1SmokeEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1SmokePPORunnerCfg",
    },
)
gym.register(
    id="myrl/Locomotion-Flat-G1Smoke-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1SmokeEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1SmokePPORunnerCfg",
    },
)
