"""myrl.core.databus — 可选的 channel-based pub/sub 数据总线。

未启用时 get_databus() 返回 None，所有 publish 站点短路，零开销。
通过 enable_databus() 或 MYRL_OSCILLOSCOPE=1 环境变量启用。
"""

from myrl.core.databus.bus import (
    DataBus,
    auto_enable_databus,
    disable_databus,
    enable_databus,
    get_databus,
)
from myrl.core.databus.channel import ChannelInfo
from myrl.core.databus.tap import Tap

__all__ = [
    "DataBus",
    "Tap",
    "ChannelInfo",
    "get_databus",
    "enable_databus",
    "disable_databus",
    "auto_enable_databus",
]
