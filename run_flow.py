import argparse
import asyncio
import os
import time
from pathlib import Path

from app.agent.cot import CoTAgent
from app.agent.manus import Manus
from app.agent.planning import PlanningAgent
from app.config import config
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import get_agent_logger, logger
from app.utils import ensure_task_dir, get_timestamp_dir


def read_prompt_from_file(file_path: str) -> str:
    """从文件中读取提示词"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading prompt file: {str(e)}")
        return ""


async def run_flow(prompt: str = None):
    # 创建时间戳目录
    task_dir = get_timestamp_dir()
    task_path = ensure_task_dir(os.path.join(config.workspace_root, task_dir))
    logger.info(f"Created task directory: {task_path}")

    # 创建多个智能体
    logger.info("Initializing agents...")
    agents = {
        "planner": PlanningAgent(),  # 负责任务规划
        "executor": Manus(),  # 负责执行具体任务
        "reviewer": CoTAgent(),  # 负责审查和验证
    }

    # 为每个智能体创建专门的日志记录器
    agent_loggers = {name: get_agent_logger(name) for name in agents.keys()}

    try:
        # 如果没有提供提示词，则从用户输入获取
        if not prompt:
            prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        logger.info("Creating planning flow...")
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
            primary_agent_key="planner",  # 设置规划智能体为主智能体
        )
        logger.info("Starting task execution...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 minute timeout for the entire execution
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Task completed in {elapsed_time:.2f} seconds")
            logger.info("Final result:")
            logger.info(result)
        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.exception("Detailed error information:")
    finally:
        logger.info(f"Task directory: {task_path}")


def main():
    parser = argparse.ArgumentParser(description="Run OpenManus with a prompt")
    parser.add_argument(
        "--prompt-file", type=str, help="Path to a file containing the prompt"
    )
    args = parser.parse_args()

    prompt = None
    if args.prompt_file:
        prompt = read_prompt_from_file(args.prompt_file)

    asyncio.run(run_flow(prompt))


if __name__ == "__main__":
    main()
