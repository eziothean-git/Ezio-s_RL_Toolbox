import subprocess, sys

def run(cmd):
    return subprocess.check_output(cmd, shell=True, text=True).strip()

print("PY:", sys.executable)

print("\n== GPU ==")
try:
    print(run("nvidia-smi -L"))
except Exception as e:
    print("nvidia-smi not available:", e)

print("\n== IsaacGym ==")
import isaacgym
from isaacgym import gymapi
print("isaacgym:", isaacgym.__file__)
print("gymapi:", gymapi)
print("\n== Torch ==")

import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(256, 256, device="cuda")
    y = torch.randn(256, 256, device="cuda")
    _ = (x @ y).mean()
    torch.cuda.synchronize()
    print("gpu:", torch.cuda.get_device_name(0))
    print("ok: kernel launch")

print("\n== Third Party ==")
import rsl_rl, legged_gym
print("rsl_rl:", rsl_rl.__file__)
print("legged_gym:", legged_gym.__file__)

print("\nSMOKE: OK")
