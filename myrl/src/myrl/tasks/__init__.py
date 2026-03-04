"""myrl 任务注册入口。

所有子目录的 config/__init__.py 会在此被自动导入（触发 gym.register）。
目前直接复用 instinctlab 的 import_packages 工具递归触发子包注册。
"""

from isaaclab_tasks.utils import import_packages

import_packages(__name__, blacklist_pkgs=["utils"])
