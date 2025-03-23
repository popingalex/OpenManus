import os
import time
from datetime import datetime
from pathlib import Path


def get_timestamp_dir() -> str:
    """生成一个基于时间戳的目录名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"task_{timestamp}"


def ensure_task_dir(task_dir: str) -> str:
    """确保任务目录存在，如果不存在则创建"""
    task_path = Path(task_dir)
    if not task_path.exists():
        task_path.mkdir(parents=True)
    return str(task_path)
