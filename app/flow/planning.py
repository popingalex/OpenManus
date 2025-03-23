import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool
from app.utils import ensure_task_dir, get_timestamp_dir


class PlanStepStatus(str, Enum):
    """Enum class defining possible statuses of a plan step"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """Return a list of all possible step status values"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """Return a list of values representing active statuses (not started or in progress)"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """Return a mapping of statuses to their marker symbols"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """A flow that manages planning and execution of tasks using agents."""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None
    task_dir: Optional[str] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # Set executor keys before super().__init__
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # Set plan ID if provided
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # Initialize the planning tool if not provided
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # Call parent's init with the processed data
        super().__init__(agents, **data)

        # Set executor_keys to all agent keys if not specified
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

        # Create task directory
        self._create_task_directory()

        # Log agent initialization
        for name, agent in self.agents.items():
            self.agent_loggers[name].info(
                f"Agent {name} initialized with type {type(agent).__name__}"
            )

    def _create_task_directory(self):
        """创建任务目录并初始化必要的文件"""
        try:
            # 创建时间戳目录
            task_dir = get_timestamp_dir()
            self.task_dir = ensure_task_dir(os.path.join("workspace", task_dir))

            # 创建计划文件
            plan_file = Path(self.task_dir) / "plan.json"
            if not plan_file.exists():
                plan_file.write_text("{}", encoding="utf-8")

            # 创建步骤文件
            steps_file = Path(self.task_dir) / "steps.json"
            if not steps_file.exists():
                steps_file.write_text("[]", encoding="utf-8")

            # 创建结果文件
            result_file = Path(self.task_dir) / "result.txt"
            if not result_file.exists():
                result_file.write_text("", encoding="utf-8")

            logger.info(f"📁 Created task directory and files at: {self.task_dir}")
        except Exception as e:
            logger.error(f"❌ Error creating task directory: {str(e)}")
            raise

    async def _save_step_result(self, step_index: int, step_info: dict, result: str):
        """保存步骤执行结果"""
        try:
            if not self.task_dir:
                return

            # 更新步骤文件
            steps_file = Path(self.task_dir) / "steps.json"
            steps_data = []
            if steps_file.exists():
                steps_data = json.loads(steps_file.read_text(encoding="utf-8"))

            # 添加或更新步骤结果
            step_data = {
                "index": step_index,
                "info": step_info,
                "result": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 更新或添加步骤
            found = False
            for i, step in enumerate(steps_data):
                if step["index"] == step_index:
                    steps_data[i] = step_data
                    found = True
                    break
            if not found:
                steps_data.append(step_data)

            # 保存更新后的步骤数据
            steps_file.write_text(
                json.dumps(steps_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            # 更新结果文件
            result_file = Path(self.task_dir) / "result.txt"
            with result_file.open("a", encoding="utf-8") as f:
                f.write(f"\n{'='*20} Step {step_index} {'='*20}\n")
                f.write(f"⏰ Time: {step_data['timestamp']}\n")
                f.write(f"📝 Info: {json.dumps(step_info, ensure_ascii=False)}\n")
                f.write(f"📊 Result:\n{result}\n")
                f.write("=" * 50 + "\n")

            logger.info(f"💾 Saved step {step_index} result to {self.task_dir}")
        except Exception as e:
            logger.error(f"❌ Error saving step result: {str(e)}")

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """执行单个步骤并保存结果"""
        try:
            # 检查是否已经执行过这个步骤
            if not hasattr(self, "_executed_steps"):
                self._executed_steps = set()

            step_key = f"{step_info.get('text', '')}_{step_info.get('type', '')}"
            if step_key in self._executed_steps:
                logger.warning(f"⚠️ Step already executed: {step_key}")
                return "Step already executed"

            # 准备步骤执行的上下文
            step_text = step_info.get("text", "")
            step_type = step_info.get("type", "").lower()

            # 使用更简短的提示
            step_prompt = f"执行：{step_text}"

            # 使用 run 方法执行步骤
            try:
                # 检查 executor 是否有效
                if not executor:
                    raise ValueError("Invalid executor")

                # 执行步骤
                step_result = await executor.run(step_prompt)

                # 检查结果是否有效
                if not step_result:
                    raise ValueError("Empty step result")

                # 检查结果是否包含有意义的内容
                if step_result.strip() == "Thinking complete - no action needed":
                    raise ValueError("No meaningful action taken")

                # 检查结果是否包含终止信号
                if (
                    "terminate" in step_result.lower()
                    or "finished" in step_result.lower()
                ):
                    logger.info("✅ Step completed with termination signal")
                    return step_result

            except Exception as e:
                if "TokenLimitExceeded" in str(e):
                    logger.warning(
                        "⚠️ Token limit exceeded, retrying with minimal context"
                    )
                    # 使用最小化的提示重试
                    minimal_prompt = step_text
                    try:
                        step_result = await executor.run(minimal_prompt)
                    except Exception as e2:
                        logger.error(f"❌ Error in minimal context: {str(e2)}")
                        raise
                else:
                    raise

            # 标记步骤为已完成
            try:
                mark_step_args = {
                    "command": "mark_step",
                    "plan_id": self.active_plan_id,
                    "step_index": self.current_step_index,
                    "step_status": PlanStepStatus.COMPLETED.value,
                    "step_notes": f"完成步骤：{step_text}",
                }
                await self.planning_tool.execute(**mark_step_args)
            except Exception as e:
                logger.warning(f"⚠️ Failed to mark step as completed: {e}")
                # 直接更新步骤状态
                if self.active_plan_id in self.planning_tool.plans:
                    plan_data = self.planning_tool.plans[self.active_plan_id]
                    step_statuses = plan_data.get("step_statuses", [])
                    while len(step_statuses) <= self.current_step_index:
                        step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                    step_statuses[self.current_step_index] = (
                        PlanStepStatus.COMPLETED.value
                    )
                    plan_data["step_statuses"] = step_statuses

            # 记录已执行的步骤
            self._executed_steps.add(step_key)

            # 保存步骤结果
            try:
                await self._save_step_result(
                    self.current_step_index, step_info, step_result
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to save step result: {e}")

            return step_result
        except Exception as e:
            error_msg = f"❌ Error executing step: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _mark_step_completed(self) -> None:
        """标记当前步骤为已完成"""
        if self.current_step_index is None:
            return

        try:
            # 标记步骤为已完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(f"✅ Marked step {self.current_step_index} as completed")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update plan status: {e}")
            # 直接在规划工具存储中更新步骤状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保 step_statuses 列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        Get an appropriate executor agent for the current step.
        Can be extended to select agents based on step type/requirements.
        """
        # Log agent selection process
        self.agent_loggers[self.primary_agent_key].info(
            f"Selecting executor for step type: {step_type}"
        )

        # If step type is provided and matches an agent key, use that agent
        if step_type and step_type in self.agents:
            self.agent_loggers[self.primary_agent_key].info(
                f"Selected agent {step_type} based on step type"
            )
            return self.agents[step_type]

        # Otherwise use the first available executor or fall back to primary agent
        for key in self.executor_keys:
            if key in self.agents:
                self.agent_loggers[self.primary_agent_key].info(
                    f"Selected agent {key} from executor keys"
                )
                return self.agents[key]

        # Fallback to primary agent
        self.agent_loggers[self.primary_agent_key].info(
            f"Falling back to primary agent {self.primary_agent_key}"
        )
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """Execute the planning flow with agents."""
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # 记录执行开始
            self.agent_loggers[self.primary_agent_key].info("=" * 50)
            self.agent_loggers[self.primary_agent_key].info(
                "🚀 Starting Planning Flow Execution"
            )
            self.agent_loggers[self.primary_agent_key].info(
                f"📝 Input Text: {input_text}"
            )
            self.agent_loggers[self.primary_agent_key].info("=" * 50)

            # Create initial plan if input provided
            if input_text:
                self.agent_loggers[self.primary_agent_key].info(
                    "📋 Creating Initial Plan..."
                )
                await self._create_initial_plan(input_text)

                # Verify plan was created successfully
                if self.active_plan_id not in self.planning_tool.plans:
                    error_msg = f"❌ Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    self.agent_loggers[self.primary_agent_key].error(error_msg)
                    return error_msg
                self.agent_loggers[self.primary_agent_key].info(
                    "✅ Initial plan created successfully"
                )

            result = ""
            step_count = 0
            max_steps = 20  # 最大步骤数限制
            consecutive_errors = 0  # 连续错误计数
            max_consecutive_errors = 3  # 最大连续错误数
            last_step_type = None  # 记录上一个步骤的类型
            same_step_type_count = 0  # 记录相同类型步骤的连续次数

            while step_count < max_steps:
                # Get current step to execute
                self.current_step_index, step_info = await self._get_current_step_info()
                step_count += 1

                # Log step information
                if step_info:
                    self.agent_loggers[self.primary_agent_key].info("-" * 30)
                    self.agent_loggers[self.primary_agent_key].info(
                        f"🔄 Step {step_count}: {step_info.get('text', '')}"
                    )
                    if "type" in step_info:
                        self.agent_loggers[self.primary_agent_key].info(
                            f"📌 Type: {step_info['type']}"
                        )
                    self.agent_loggers[self.primary_agent_key].info("-" * 30)

                # Exit if no more steps or plan completed
                if self.current_step_index is None:
                    self.agent_loggers[self.primary_agent_key].info("=" * 50)
                    self.agent_loggers[self.primary_agent_key].info(
                        "🏁 No more steps to execute"
                    )
                    self.agent_loggers[self.primary_agent_key].info("=" * 50)
                    result += await self._finalize_plan()
                    break

                # 检查步骤类型是否重复
                current_step_type = step_info.get("type") if step_info else None
                if current_step_type == last_step_type:
                    same_step_type_count += 1
                    if same_step_type_count >= 3:  # 如果连续执行相同类型的步骤超过3次
                        self.agent_loggers[self.primary_agent_key].warning(
                            f"⚠️ Too many consecutive steps of type {current_step_type}. Stopping execution."
                        )
                        break
                else:
                    same_step_type_count = 0
                    last_step_type = current_step_type

                # Execute current step with appropriate agent
                executor = self.get_executor(current_step_type)

                # Log agent execution
                self.agent_loggers[self.primary_agent_key].info(
                    f"🤖 Executing with agent: {executor.name}"
                )
                try:
                    step_result = await self._execute_step(executor, step_info)
                    result += step_result + "\n"
                    consecutive_errors = 0  # 重置连续错误计数

                    # 检查结果是否包含终止信号
                    if (
                        "terminate" in step_result.lower()
                        or "finished" in step_result.lower()
                    ):
                        self.agent_loggers[self.primary_agent_key].info(
                            "✅ Received termination signal from step execution"
                        )
                        break

                except Exception as e:
                    error_msg = f"❌ Error executing step: {str(e)}"
                    self.agent_loggers[self.primary_agent_key].error(error_msg)
                    result += error_msg + "\n"
                    consecutive_errors += 1

                    # 如果连续错误次数过多，终止执行
                    if consecutive_errors >= max_consecutive_errors:
                        self.agent_loggers[self.primary_agent_key].error(
                            f"❌ Too many consecutive errors ({consecutive_errors}). Stopping execution."
                        )
                        break

                # Check if agent wants to terminate
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    self.agent_loggers[self.primary_agent_key].info(
                        f"✅ Agent {executor.name} finished execution"
                    )
                    # 使用正确的参数格式调用终止命令
                    try:
                        await executor.run(
                            {"command": "terminate", "status": "success"}
                        )
                    except Exception as e:
                        logger.error(f"Error terminating execution: {e}")
                    break

            # 如果达到最大步骤数，记录警告
            if step_count >= max_steps:
                self.agent_loggers[self.primary_agent_key].warning(
                    f"⚠️ Reached maximum step limit ({max_steps}). Stopping execution."
                )

            # 记录执行完成
            self.agent_loggers[self.primary_agent_key].info("=" * 50)
            self.agent_loggers[self.primary_agent_key].info(
                f"✨ Planning Flow Completed - Total Steps: {step_count}"
            )
            self.agent_loggers[self.primary_agent_key].info("=" * 50)

            return result
        except Exception as e:
            error_msg = f"❌ Error in PlanningFlow: {str(e)}"
            self.agent_loggers[self.primary_agent_key].error(error_msg)
            return error_msg

    async def _create_initial_plan(self, request: str) -> None:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        try:
            # 创建默认计划
            default_plan = {
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": [
                    "理解需求",
                    "确定目标",
                    "制定方案",
                    "评估可行性",
                    "实施方案",
                    "处理问题",
                    "检查结果",
                    "总结反馈",
                ],
            }

            # 检查计划是否已存在
            if self.active_plan_id in self.planning_tool.plans:
                logger.info(f"Plan {self.active_plan_id} already exists, updating...")
                default_plan["command"] = "update"
                # 保留现有的步骤状态
                existing_plan = self.planning_tool.plans[self.active_plan_id]
                default_plan["step_statuses"] = existing_plan.get("step_statuses", [])
                default_plan["step_notes"] = existing_plan.get("step_notes", [])

            # 确保计划被创建或更新
            await self.planning_tool.execute(**default_plan)
            logger.info(f"Created/Updated plan with ID: {self.active_plan_id}")

            # 创建系统消息，限制 token 使用
            system_message = Message.system_message(
                "你是一个规划助手。请创建一个简洁、可执行的计划。"
                "保持步骤清晰且易于理解。"
                "优化响应长度，避免冗余信息。"
            )

            # 创建用户消息，限制输入长度
            user_message = Message.user_message(
                f"请为以下任务创建实施计划：{request[:200]}"
            )

            # 调用 LLM 并处理响应
            try:
                response = await self.llm.ask_tool(
                    messages=[user_message],
                    system_msgs=[system_message],
                    tools=[self.planning_tool.to_param()],
                    tool_choice=ToolChoice.AUTO,
                )

                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "planning":
                            try:
                                # 解析参数
                                args = tool_call.function.arguments
                                if isinstance(args, str):
                                    args = json.loads(args)

                                # 确保 plan_id 正确
                                args["plan_id"] = self.active_plan_id

                                # 如果计划已存在，使用 update 命令
                                if self.active_plan_id in self.planning_tool.plans:
                                    args["command"] = "update"
                                    # 保留现有的步骤状态
                                    existing_plan = self.planning_tool.plans[
                                        self.active_plan_id
                                    ]
                                    args["step_statuses"] = existing_plan.get(
                                        "step_statuses", []
                                    )
                                    args["step_notes"] = existing_plan.get(
                                        "step_notes", []
                                    )

                                # 验证参数格式
                                if not isinstance(args, dict):
                                    raise ValueError("Arguments must be a dictionary")

                                # 执行工具调用
                                result = await self.planning_tool.execute(**args)
                                logger.info(
                                    f"Updated plan with LLM suggestions: {str(result)}"
                                )
                                return
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse tool arguments: {e}")
                                continue
                            except Exception as e:
                                logger.error(f"Error executing planning tool: {e}")
                                continue

            except Exception as e:
                logger.error(f"Error in LLM call: {e}")
                # 继续使用默认计划

            logger.info("Using default plan as no valid LLM suggestions were received")

        except Exception as e:
            logger.error(f"Error in _create_initial_plan: {e}")
            # 确保至少有一个默认计划
            try:
                if self.active_plan_id in self.planning_tool.plans:
                    default_plan["command"] = "update"
                    # 保留现有的步骤状态
                    existing_plan = self.planning_tool.plans[self.active_plan_id]
                    default_plan["step_statuses"] = existing_plan.get(
                        "step_statuses", []
                    )
                    default_plan["step_notes"] = existing_plan.get("step_notes", [])
                await self.planning_tool.execute(**default_plan)
                logger.info(
                    f"Created/Updated fallback plan with ID: {self.active_plan_id}"
                )
            except Exception as e2:
                logger.error(f"Failed to create/update fallback plan: {e2}")
                raise

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        Parse the current plan to identify the first non-completed step's index and info.
        Returns (None, None) if no active step is found.
        """
        try:
            if not self.active_plan_id:
                logger.error("No active plan ID")
                return None, None

            if self.active_plan_id not in self.planning_tool.plans:
                logger.error(f"Plan with ID {self.active_plan_id} not found")
                return None, None

            # 直接访问规划工具存储中的计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            if not plan_data:
                logger.error(f"Plan data is empty for ID {self.active_plan_id}")
                return None, None

            steps = plan_data.get("steps", [])
            if not steps:
                logger.error(f"No steps found in plan {self.active_plan_id}")
                return None, None

            # 确保 steps 是有效的字符串列表
            steps = [str(step) for step in steps if step]
            if not steps:
                logger.error("No valid steps found in plan")
                return None, None

            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保 step_statuses 和 step_notes 列表足够长
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if not step or not isinstance(step, str):
                    logger.warning(f"Invalid step at index {i}: {step}")
                    continue

                status = step_statuses[i]
                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤类型/类别
                    step_text = step.lower()
                    step_type = None

                    # 使用更精确的关键词匹配
                    analyze_keywords = ["分析", "分析需求", "制定计划", "规划", "设计"]
                    execute_keywords = ["执行", "实现", "创建", "编写", "开发", "构建"]
                    verify_keywords = ["验证", "测试", "检查", "确认", "评估"]

                    if any(keyword in step_text for keyword in analyze_keywords):
                        step_type = "analyze"
                    elif any(keyword in step_text for keyword in execute_keywords):
                        step_type = "execute"
                    elif any(keyword in step_text for keyword in verify_keywords):
                        step_type = "verify"
                    else:
                        # 如果没有匹配到关键词，根据步骤位置分配类型
                        if i == 0:
                            step_type = "analyze"
                        elif i == len(steps) - 1:
                            step_type = "verify"
                        else:
                            step_type = "execute"

                    step_info = {
                        "text": step,
                        "type": step_type,
                        "index": i,
                        "status": status,
                        "notes": step_notes[i] if i < len(step_notes) else "",
                    }

                    # 标记当前步骤为进行中
                    try:
                        mark_step_args = {
                            "command": "mark_step",
                            "plan_id": self.active_plan_id,
                            "step_index": i,
                            "step_status": PlanStepStatus.IN_PROGRESS.value,
                            "step_notes": f"开始执行步骤：{step}",
                        }
                        await self.planning_tool.execute(**mark_step_args)
                    except Exception as e:
                        logger.warning(f"Error marking step as in_progress: {e}")
                        # 直接更新步骤状态
                        step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            logger.info("No active steps found in plan")
            return None, None

        except Exception as e:
            logger.error(f"Error in _get_current_step_info: {e}")
            return None, None

    async def _get_plan_text(self) -> str:
        """Get the current plan as formatted text."""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """Generate plan text directly from storage if the planning tool fails."""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # Ensure step_statuses and step_notes match the number of steps
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # Count steps by status
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # Use status marks to indicate step status
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """Finalize the plan and provide a summary using the flow's LLM directly."""
        plan_text = await self._get_plan_text()

        # Create a summary using the flow's LLM directly
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # Fallback to using an agent for the summary
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
