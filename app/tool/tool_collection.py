"""Collection classes for managing multiple tools."""

import json
from typing import Any, Dict, List, Union

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """A collection of defined tools."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Union[str, Dict[str, Any]] = None
    ) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")

        try:
            # 如果输入是字符串，尝试解析为字典
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    return ToolFailure(error=f"Invalid JSON input for tool {name}")

            # 确保 tool_input 是字典类型
            if not isinstance(tool_input, dict):
                return ToolFailure(error=f"Tool {name} expects a dictionary input")

            # 验证必需参数
            required_params = tool.to_param().get("parameters", {}).get("required", [])
            missing_params = [
                param for param in required_params if param not in tool_input
            ]
            if missing_params:
                return ToolFailure(
                    error=f"Missing required parameters for tool {name}: {', '.join(missing_params)}"
                )

            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)
        except Exception as e:
            return ToolFailure(error=f"Error executing tool {name}: {str(e)}")

    async def execute_all(self) -> List[ToolResult]:
        """Execute all tools in the collection sequentially."""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        for tool in tools:
            self.add_tool(tool)
        return self
