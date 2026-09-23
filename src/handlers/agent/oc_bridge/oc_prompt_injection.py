"""
OC-specific prompt injection helpers — drain background task notifications
and pending approval items into the agent's working memory so the LLM
sees them as new input.

Extracted from ``ChatAgentHandler._build_prompt_input`` to keep the main
agent file free of OC-specific prompt-building logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from handlers.agent.memory.session_memory_manager import SessionMemoryManager
    from handlers.agent.oc_bridge.pending_confirmations import PendingConfirmationsManager
    from handlers.agent.oc_bridge.task_notification_queue import TaskNotificationQueue


def drain_task_notifications(
    task_queue: Optional["TaskNotificationQueue"],
    memory: Optional["SessionMemoryManager"],
) -> None:
    """Drain merged notifications from *task_queue* into *memory* as a
    ``user``-role turn so the LLM treats them as new background input."""
    if not task_queue:
        return
    notifications = task_queue.drain_merged()
    if notifications and memory:
        notif_text = task_queue.format_for_prompt(notifications)
        memory.working_memory.add_user_turn(
            content=notif_text, trigger_type="background"
        )


def inject_pending_approvals(
    pending_mgr: Optional["PendingConfirmationsManager"],
    memory: Optional["SessionMemoryManager"],
) -> None:
    """If there are pending approval items, inject a ``<pending-approvals>``
    block into *memory* so the agent always has the approval_id available —
    even if the original ``<background-results>`` was drained earlier."""
    if not pending_mgr or not pending_mgr.has_pending() or not memory:
        return

    pending_items = pending_mgr.get_pending_items()
    pa_lines = []
    for item in pending_items:
        pa_lines.append(
            f"- approval_id=\"{item.id}\" | {item.text}\n"
            f"  → Call exec_approve(approval_id=\"{item.id}\", "
            f"decision=<allow-once|allow-always|deny>)"
        )
    pa_text = (
        "<pending-approvals>\n"
        "The following approvals are still pending. Stating 'Approved' verbally has NO effect; "
        "you MUST invoke the exec_approve tool for approval to take effect.\n"
        "If similar approvals recur multiple times, suggest the user select allow-always:\n"
        + "\n".join(pa_lines)
        + "\n</pending-approvals>"
    )
    memory.working_memory.add_user_turn(
        content=pa_text, trigger_type="background"
    )

