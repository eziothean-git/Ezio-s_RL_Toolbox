import gymnasium as gym
from . import agents
from .env_cfg import G1CustomEnvCfg, G1CustomEnvCfg_PLAY

gym.register(
    id="myrl/Locomotion-Flat-G1Custom-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1CustomEnvCfg,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1CustomPPORunnerCfg",
    },
)
gym.register(
    id="myrl/Locomotion-Flat-G1Custom-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1CustomEnvCfg_PLAY,
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1CustomPPORunnerCfg",
    },
)
