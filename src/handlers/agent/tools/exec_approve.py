"""
ExecApproveTool — lets the main agent approve or deny OC exec requests.

When OC needs approval for a command execution, the notification arrives via
TaskNotificationQueue.  The agent asks the user, then calls this tool to
relay the decision back to OC via the oac-bridge channel.

On success the tool automatically marks the corresponding
PendingConfirmationsManager entry as resolved so nag reminders and
proactive triggers stop firing for that approval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from handlers.agent.tools.base_tool import BaseTool, ToolResult

if TYPE_CHECKING:
    from handlers.agent.oc_bridge.pending_confirmations import PendingConfirmationsManager


class ExecApproveTool(BaseTool):
    """Approve or deny an OpenClaw exec-approval request."""

    def __init__(
        self,
        oc_channel_client=None,
        oac_session_id: str = "",
        pending_mgr: PendingConfirmationsManager | None = None,
    ):
        self._oc_channel_client = oc_channel_client
        self._oac_session_id = oac_session_id
        self._pending_mgr = pending_mgr

    @property
    def name(self) -> str:
        return "exec_approve"

    @property
    def description(self) -> str:
        return (
            "Approve or deny command execution approval requests from OpenClaw backend.\n"
            "When pending approval_ids are listed in <pending-approvals>, "
            "it indicates backend commands are waiting for your authorization.\n"
            "You must first inform the user of the command details in spoken words and ask for consent, "
            "then immediately call this tool once the user provides clear instructions.\n\n"
            "Three options for decision:\n"
            "- \"allow-once\"  — Allow this single execution only\n"
            "- \"allow-always\" — Always allow this type of command (no future prompt needed)\n"
            "- \"deny\"        — Reject execution\n\n"
            "When similar approval requests recur multiple times in the same task, "
            "proactively inform the user they can choose \"allow-always\" to reduce repetitive prompts.\n"
            "IMPORTANT: Approvals do NOT take effect without invoking this tool! Verbal confirmation alone is ineffective."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "approval_id": {
                    "type": "string",
                    "description": "ID of pending approval request (from exec.approval.requested notification)",
                },
                "decision": {
                    "type": "string",
                    "enum": ["allow-once", "allow-always", "deny"],
                    "description": "Approval decision: allow-once (single use), allow-always (persistently allow), deny (reject)",
                },
            },
            "required": ["approval_id", "decision"],
        }

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        approval_id = args.get("approval_id", "").strip()
        decision = args.get("decision", "").strip()

        if not approval_id:
            return ToolResult(success=False, error="approval_id is required")
        if decision not in ("allow-once", "allow-always", "deny"):
            return ToolResult(
                success=False,
                error=f"Invalid decision: {decision}. Must be allow-once, allow-always, or deny",
            )
        if not self._oc_channel_client:
            return ToolResult(success=False, error="OC channel client not available")

        approve_text = f"/approve {approval_id} {decision}"
        logger.info(
            f"[ExecApprove] Sending approval: {approve_text} "
            f"(session={self._oac_session_id})"
        )

        result = self._oc_channel_client.send_message(
            oac_session_id=self._oac_session_id,
            text=approve_text,
            sender_name="OAC Agent",
        )

        if "error" in result:
            return ToolResult(
                success=False,
                error=f"Failed to send approval: {result['error']}",
            )

        decision_label = {
            "allow-once": "Approved (single use)",
            "allow-always": "Approved (always allow)",
            "deny": "Denied",
        }.get(decision, decision)

        if self._pending_mgr:
            resolved_status = "denied" if decision == "deny" else "confirmed"
            try:
                self._pending_mgr.upsert(
                    [{"id": approval_id, "status": resolved_status}]
                )
            except Exception as e:
                logger.warning(f"[ExecApprove] Failed to update pending item: {e}")

        return ToolResult(
            success=True,
            data={
                "approval_id": approval_id,
                "decision": decision,
                "message": f"Approval result sent: {decision_label}",
            },
        )
