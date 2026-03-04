from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)
import instinctlab.tasks.locomotion.mdp as instinct_mdp

from myrl.core.compat.views import RobotHandle, make_term


# ── 自定义 obs 函数（使用 RobotHandle，不直接读 tensor） ──────────────────

def _base_ang_vel(robot: RobotHandle):
    return robot.root_ang_vel_b                    # (num_envs, 3)

def _base_lin_vel(robot: RobotHandle):
    return robot.root_lin_vel_b                    # (num_envs, 3)

def _projected_gravity(robot: RobotHandle):
    return robot.projected_gravity_b               # (num_envs, 3)

def _joint_pos_rel(robot: RobotHandle):
    return robot.joints.pos_rel                    # (num_envs, 29)

def _joint_vel(robot: RobotHandle):
    return robot.joints.vel                        # (num_envs, 29)

def _velocity_commands(robot: RobotHandle):
    return robot.get_command("base_velocity")[:, :3]  # (num_envs, 3)


# ── Policy obs 配置（替换 G1FlatEnvCfg 中的 obs terms） ─────────────────

@configclass
class G1CustomPolicyObsCfg(ObsGroup):
    """使用 myrl Views 重写的 policy obs，维度与 g1_smoke 相同（policy=96, critic=99）。

    不加 history_length 以匹配 G1FlatEnvCfg 基线维度：
      base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
      + joint_pos(29) + joint_vel(29) + actions(29) = 96
    """

    base_ang_vel      = ObsTerm(func=make_term(_base_ang_vel))
    projected_gravity = ObsTerm(func=make_term(_projected_gravity))
    velocity_commands = ObsTerm(func=make_term(_velocity_commands))
    joint_pos         = ObsTerm(func=make_term(_joint_pos_rel))
    joint_vel         = ObsTerm(func=make_term(_joint_vel))
    # 有状态项（action history）继续复用 instinctlab mdp
    actions           = ObsTerm(func=instinct_mdp.last_action)


@configclass
class G1CustomEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = G1CustomPolicyObsCfg()


@configclass
class G1CustomEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = G1CustomPolicyObsCfg()
