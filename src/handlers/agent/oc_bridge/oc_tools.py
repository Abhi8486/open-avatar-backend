"""
OC Tools — adapts OpenClaw MCP tools to BaseTool for ToolRegistry.

These tools are exposed to the LLM so it can autonomously decide when
to search memory, retrieve agent profile, or list scheduled tasks.

All tools go through Plugin Tools MCP (no direct file reads).
"""

from typing import Any, Dict

from loguru import logger

from handlers.agent.tools.base_tool import BaseTool, ToolResult
from handlers.agent.oc_bridge.mcp_client import OcMcpClient


class OcMemorySearchTool(BaseTool):
    """Search OpenClaw's long-term memory (MEMORY.md + vector store)."""

    def __init__(self, mcp_client: OcMcpClient):
        self._client = mcp_client

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search long-term memory. Use when the user mentions past events, preferences, "
            "prior agreements, historical events, or any context requiring recall of past dialogue."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what you want to recall",
                },
                "maxResults": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query"],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self._client.is_available:
            return ToolResult(success=False, error="OpenClaw not available")

        result = self._client.call_tool_sync("memory_search", args, timeout=15.0)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)


class OcMemoryGetTool(BaseTool):
    """Read a specific snippet from OpenClaw's memory files (MEMORY.md / memory/*.md)."""

    def __init__(self, mcp_client: OcMcpClient):
        self._client = mcp_client

    @property
    def name(self) -> str:
        return "memory_get"

    @property
    def description(self) -> str:
        return (
            "Read a specific snippet from long-term memory files. Use memory_search first "
            "to locate the file path and line numbers, then call this tool to fetch exact text."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to memory file, e.g. MEMORY.md or memory/notes.md",
                },
                "from": {
                    "type": "integer",
                    "description": "Start line number (optional, default start of file)",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read (optional, default all)",
                },
            },
            "required": ["path"],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self._client.is_available:
            return ToolResult(success=False, error="OpenClaw not available")

        result = self._client.call_tool_sync("memory_get", args, timeout=15.0)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)


class OcGetAgentProfileTool(BaseTool):
    """Get agent identity, personality, and user preferences from OC workspace via MCP."""

    def __init__(self, mcp_client: OcMcpClient):
        self._client = mcp_client

    @property
    def name(self) -> str:
        return "get_agent_profile"

    @property
    def description(self) -> str:
        return (
            "Get agent identity information and user preferences. Use when uncertain about "
            "how to address the user, your own persona settings, or user preferences."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Sections to fetch: "identity", "soul", "user". Defaults to all.',
                },
            },
            "required": [],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self._client.is_available:
            return ToolResult(success=False, error="OpenClaw not available")

        result = self._client.call_tool_sync("get_agent_profile", args, timeout=15.0)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)


class OcListScheduledTasksTool(BaseTool):
    """List active scheduled tasks (cron jobs) from OpenClaw via MCP."""

    def __init__(self, mcp_client: OcMcpClient):
        self._client = mcp_client

    @property
    def name(self) -> str:
        return "list_scheduled_tasks"

    @property
    def description(self) -> str:
        return (
            "View configured scheduled tasks and routines. Use when user asks 'What tasks were assigned to you?', "
            "about scheduled reminders, routine events, or checking task execution status."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "include_disabled": {
                    "type": "boolean",
                    "description": "Whether to include disabled tasks (default false)",
                },
            },
            "required": [],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if not self._client.is_available:
            return ToolResult(success=False, error="OpenClaw not available")

        result = self._client.call_tool_sync("list_scheduled_tasks", args, timeout=15.0)
        if "error" in result:
            return ToolResult(success=False, error=result["error"])
        return ToolResult(success=True, data=result)


def register_oc_tools(registry, mcp_client: OcMcpClient):
    """Register available OC tools into a ToolRegistry.

    Probes Plugin Tools MCP to discover which tools are available.
    """
    if not mcp_client.is_available:
        logger.warning("[OcTools] MCP not available, skipping tool registration")
        return

    plugin_tools = mcp_client.list_tools_sync()
    plugin_tool_names = {t["name"] for t in plugin_tools} if plugin_tools else set()

    registered = []
    if "memory_search" in plugin_tool_names:
        registry.register(OcMemorySearchTool(mcp_client))
        registered.append("memory_search")
    if "memory_get" in plugin_tool_names:
        registry.register(OcMemoryGetTool(mcp_client))
        registered.append("memory_get")
    if "get_agent_profile" in plugin_tool_names:
        registry.register(OcGetAgentProfileTool(mcp_client))
        registered.append("get_agent_profile")
    if "list_scheduled_tasks" in plugin_tool_names:
        registry.register(OcListScheduledTasksTool(mcp_client))
        registered.append("list_scheduled_tasks")

    if registered:
        logger.info(f"[OcTools] Registered plugin tools: {registered}")
    else:
        logger.warning(
            "[OcTools] Plugin Tools MCP connected but no tools found. "
            "Check that plugins are loaded: openclaw plugins list"
        )
