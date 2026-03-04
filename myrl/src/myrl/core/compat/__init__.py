"""myrl compat 层公开接口。

注意：IsaacLabBackend 依赖 isaaclab.envs（omni.physics 运行时），
必须在 AppLauncher 初始化 Isaac Sim 之后才能导入。
使用时直接按需导入，不要依赖此 __init__.py 的顶层 import：
    from myrl.core.compat.backends.isaaclab_backend import IsaacLabBackend
"""

__all__ = ["IsaacLabBackend"]
