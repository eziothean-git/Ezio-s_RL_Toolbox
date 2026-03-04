from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from myrl.assets import resolve_asset
from instinctlab.tasks.locomotion.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)
from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_POPSICLE_CFG

# 资产优先级：myrl/assets/ > instinctlab 内置
_MYRL_G1_URDF = resolve_asset("robots/g1/urdf/g1_29dof_torsobase_popsicle.urdf")

if _MYRL_G1_URDF is not None:
    # 使用 myrl 项目自带 G1（未来放置自定义改装版 URDF）
    _ROBOT_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG.replace(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_MYRL_G1_URDF,
            fix_base=False,
            replace_cylinders_with_capsules=True,
            activate_contact_sensors=True,
        )
    )
else:
    # 回退：使用 instinctlab 打包的 G1 URDF（smoke test / pipeline 验证用）
    _ROBOT_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG


@configclass
class G1SmokeEnvCfg(G1FlatEnvCfg):
    """myrl G1 冒烟任务 — 继承 instinctlab flat 配置，只替换机器人资产来源。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1SmokeEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play 变体（可视化用）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
