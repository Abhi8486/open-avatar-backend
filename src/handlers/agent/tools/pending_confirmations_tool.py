"""
PendingConfirmationsTool — lets the agent manage a list of items awaiting
user confirmation.

This is the tool-call interface to ``PendingConfirmationsManager``.  The LLM
writes/updates items via this tool; the manager tracks state and produces
nag reminders when items stay ``pending`` too long.
"""

from typing import Any, Dict

from loguru import logger

from handlers.agent.tools.base_tool import BaseTool, ToolResult


class PendingConfirmationsTool(BaseTool):
    """Manage the pending-confirmations list."""

    def __init__(self, manager=None):
        self._manager = manager

    @property
    def name(self) -> str:
        return "pending_confirmations"

    @property
    def description(self) -> str:
        return (
            "Manage the list of items awaiting user confirmation. "
            "Write items when notifications requiring user confirmation (e.g. command execution approval) are received, "
            "and update status after user approves or denies."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "List of pending confirmation items to create or update",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier",
                            },
                            "text": {
                                "type": "string",
                                "description": "Description text",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "confirmed", "denied", "expired"],
                                "description": "Status",
                            },
                            "source": {
                                "type": "string",
                                "description": "Source identifier, e.g. exec_approval",
                            },
                        },
                        "required": ["id"],
                    },
                },
            },
            "required": ["items"],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        if self._manager is None:
            return ToolResult(
                success=False,
                error="PendingConfirmationsManager not initialized",
            )

        items = args.get("items", [])
        if not items:
            return ToolResult(
                success=True,
                data={"list": self._manager.render()},
            )

        rendered = self._manager.upsert(items)
        logger.info(f"[PendingConfirmTool] Updated {len(items)} items")
        return ToolResult(
            success=True,
            data={"list": rendered},
        )
