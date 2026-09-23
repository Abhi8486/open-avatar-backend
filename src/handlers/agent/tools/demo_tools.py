"""
Demo tools for testing the ToolRegistry + Agent Loop pipeline.

These are simple tools to verify end-to-end tool_use flow.
Can be disabled in production via config.
"""

import platform
from datetime import datetime
from typing import Any, Dict

from handlers.agent.tools.base_tool import BaseTool, ToolResult


class GetCurrentTimeTool(BaseTool):
    """Returns the current time and date."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return (
            "Get the current date and time. Must be invoked whenever the user asks for "
            "what time it is, today's date, day of week, current time of day, or any content requiring real-time time."
            "Do not guess the time based on dialogue history, context, or model memory."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g. 'Asia/Shanghai' or 'UTC'. Defaults to system local timezone.",
                },
            },
            "required": [],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        tz_name = args.get("timezone")
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(tz_name)
                now = datetime.now(tz)
            except Exception:
                now = datetime.now()
        else:
            now = datetime.now()

        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return ToolResult(
            success=True,
            data={
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": weekdays[now.weekday()],
                "timezone": tz_name or "local",
            },
        )


class GetSystemInfoTool(BaseTool):
    """Returns basic system information."""

    @property
    def name(self) -> str:
        return "get_system_info"

    @property
    def description(self) -> str:
        return "Get basic system information (OS, hostname, etc.). Use when user asks about runtime environment or device info."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "os": platform.system(),
                "os_version": platform.version(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
            },
        )
