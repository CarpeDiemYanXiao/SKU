import json
from typing import Any

def read_json(path: str):
    """读取 json 文件"""
    with open(path) as f:
        d = json.load(f)
    return d

def save_json(data: Any, path: str, **kwargs):
    """保存 json 文件"""
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)