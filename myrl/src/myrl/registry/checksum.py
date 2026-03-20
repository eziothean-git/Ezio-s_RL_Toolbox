"""内容寻址校验和工具（stdlib only）。"""
import hashlib
import json
import os


def sha256_file(path: str) -> str:
    """计算文件的 SHA-256 十六进制摘要。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """计算字节的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(data).hexdigest()


def sha256_dict(d: dict) -> str:
    """计算字典的确定性 SHA-256（JSON 序列化后）。"""
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return sha256_bytes(canonical.encode())
