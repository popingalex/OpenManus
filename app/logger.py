import logging
import sys
from datetime import datetime
from pathlib import Path

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 创建日志文件名（使用时间戳）
log_file = log_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.log"

# 配置日志格式
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 文件处理器
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(formatter)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# 配置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# 创建智能体专用的日志记录器
def get_agent_logger(agent_name: str) -> logging.Logger:
    """为特定智能体创建日志记录器"""
    agent_logger = logging.getLogger(f"agent.{agent_name}")
    agent_logger.setLevel(logging.INFO)
    return agent_logger


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
