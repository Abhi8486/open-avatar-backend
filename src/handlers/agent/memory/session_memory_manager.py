"""
Session Memory Manager — Unified Memory Management Layer

Coordinates the four sub-systems: WorkingMemory / PerceptionBuffer / SessionSummary / WriteBackQueue,
providing a single entry point for ChatAgentHandler.

Responsibilities:
- Initialization and lifecycle management
- Cross-subsystem collaborative operations (e.g., user dialogue updating WorkingMemory and triggering SessionSummary)
- Automatic evaluation of whether perception events need write-back to OC upon ingestion
- Providing unified context queries for Prompt Compiler
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger

from handlers.agent.memory.working_memory import WorkingMemory, DialogueTurn
from handlers.agent.memory.perception_buffer import (
    PerceptionBuffer,
    PerceptionEntry,
    WRITEBACK_IMPORTANCE_THRESHOLD,
)
from handlers.agent.memory.session_summary import SessionSummaryGenerator, SessionSummary
from handlers.agent.memory.write_behind_queue import (
    WriteBackQueue,
    WriteBackItem,
    LocalWriteBackQueue,
)


@dataclass
class MemoryConfig:
    """Memory system configuration, decoupled from ChatAgentConfig."""
    # WorkingMemory
    max_dialogue_turns: int = 20

    # PerceptionBuffer
    perception_max_entries: int = 100
    perception_decay_rate: float = 0.1
    perception_decay_interval: float = 60.0
    perception_aggregation_window: float = 10.0
    perception_category_ttl: Optional[Dict[str, float]] = None

    # SessionSummary
    summary_update_interval_turns: int = 5

    # WriteBackQueue
    writeback_importance_threshold: float = WRITEBACK_IMPORTANCE_THRESHOLD
    writeback_max_queue_size: int = 1000

    # Auto-Compact (Phase 2.5 → 3.3 Enhanced)
    compact_enabled: bool = True
    compact_threshold: int = 15
    compact_keep_recent: int = 5
    compact_save_transcript: bool = True
    rehydrate_task_brief: bool = True
    rehydrate_env_state: bool = True


class SessionMemoryManager:
    """
    Unified memory manager, one instance per session.

    Usage in ChatAgentHandler:
        memory = SessionMemoryManager(config)
        memory.record_user_input("Hello", intent="greeting", perception_snapshot="...")
        memory.record_perception("Office scene", category="scene")
        memory.record_assistant_response("Hello! How can I help you today?")

        # Get context for Prompt compilation
        perception_summary = memory.get_perception_summary(seconds=120)
        dialogue_messages = memory.get_dialogue_for_llm(n=10)
        session_summary_text = memory.get_session_summary_text()
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        write_back_queue: Optional[WriteBackQueue] = None,
    ):
        self.config = config or MemoryConfig()

        self.working_memory = WorkingMemory(
            max_turns=self.config.max_dialogue_turns,
        )

        self.perception_buffer = PerceptionBuffer(
            max_entries=self.config.perception_max_entries,
            decay_rate=self.config.perception_decay_rate,
            decay_interval=self.config.perception_decay_interval,
            category_ttl=self.config.perception_category_ttl,
            aggregation_window=self.config.perception_aggregation_window,
        )

        self._summary_generator = SessionSummaryGenerator(
            update_interval_turns=self.config.summary_update_interval_turns,
        )

        self.write_back_queue = write_back_queue or LocalWriteBackQueue(
            max_queue_size=self.config.writeback_max_queue_size,
        )

        logger.info("[SessionMemoryManager] initialized")

    # ── Dialogue Recording ──

    def record_user_input(
        self,
        content: str,
        intent: Optional[str] = None,
        trigger_type: str = "user",
        perception_snapshot: Optional[str] = None,
    ):
        """Record user input and trigger summary update check."""
        self.working_memory.add_user_turn(
            content=content,
            intent=intent,
            trigger_type=trigger_type,
            perception_snapshot=perception_snapshot,
        )
        self._maybe_update_summary()

    def record_assistant_response(self, content: str):
        """Record assistant response."""
        self.working_memory.add_assistant_turn(content)
        self._maybe_update_summary()

    def record_system_event(self, content: str, trigger_type: str = "event"):
        """Record system event to dialogue history."""
        self.working_memory.add_system_turn(content, trigger_type=trigger_type)

    # ── Perception Recording ──

    def record_perception(
        self,
        content: str,
        category: str,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
        event_type: Optional[str] = None,
    ):
        """
        Record perception event.

        Automatically evaluates if OC write-back is needed:
        - Events with importance >= writeback_importance_threshold are enqueued for write-back.
        """
        self.perception_buffer.add(
            content=content,
            category=category,
            importance=importance,
            metadata=metadata,
            event_type=event_type,
        )

        if importance >= self.config.writeback_importance_threshold:
            self.write_back_queue.enqueue(WriteBackItem(
                item_type="event" if category == "event" else "episodic",
                content=content,
                importance=importance,
                metadata=metadata or {},
            ))

    # ── Context Queries (For Prompt Compiler) ──

    def get_environment_state(self, max_length: int = 500) -> str:
        """L6: Get persistent environment state snapshot (latest scene + micro-compact history)."""
        return self.perception_buffer.get_state_summary(max_length=max_length)

    def get_recent_perception_events(
        self, max_age: float = 60.0
    ) -> List[Dict]:
        """L7: Get recent discrete events formatted as dict list for <observation> injection."""
        import time as _time
        entries = self.perception_buffer.get_recent_events(max_age=max_age)
        now = _time.time()
        return [
            {
                "source": "camera",
                "time": _time.strftime("%H:%M:%S", _time.localtime(e.timestamp)),
                "content": e.content,
                "event_type": e.metadata.get("event_type", ""),
                "age_seconds": round(now - e.timestamp, 1),
            }
            for e in entries
        ]

    def get_perception_summary(
        self,
        seconds: float = 120.0,
        max_length: int = 200,
    ) -> str:
        """Get perception summary for recent time window (legacy compatible)."""
        return self.perception_buffer.get_recent_summary(
            seconds=seconds, max_length=max_length
        )

    def get_perception_full_summary(self, max_length: int = 500) -> str:
        """Get full perception summary (sorted by importance)."""
        return self.perception_buffer.get_summary(max_length=max_length)

    def get_dialogue_for_llm(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent N dialogue turns, formatted for LLM messages."""
        return self.working_memory.get_llm_messages(n)

    def get_recent_turns(self, n: Optional[int] = None) -> List[DialogueTurn]:
        """Get recent N full DialogueTurns."""
        return self.working_memory.get_recent_turns(n)

    def get_session_summary_text(self) -> str:
        """Get current session summary text."""
        return self._summary_generator.get_text()

    def get_session_summary(self) -> SessionSummary:
        """Get full SessionSummary data object."""
        return self._summary_generator.summary

    @property
    def current_intent(self) -> Optional[str]:
        return self.working_memory.current_intent

    @property
    def session_mode(self) -> str:
        return self.working_memory.session_mode

    @property
    def turn_count(self) -> int:
        return self.working_memory.turn_count

    # ── Session State Management ──

    def update_session_mode(self, mode: str):
        self.working_memory.update_session_mode(mode)

    def set_task_summary(self, summary: str):
        """Called by OC Bridge to set latest task summary."""
        self.working_memory.set_task_summary(summary)

    # ── Comprehensive Snapshot ──

    def get_full_context_snapshot(self) -> Dict:
        """
        Export complete snapshot across all memory layers for debugging or passing to PromptCompiler.
        """
        return {
            "working_memory": self.working_memory.get_context_snapshot(),
            "perception": {
                "recent_summary": self.get_perception_summary(),
                "stats": self.perception_buffer.get_stats(),
            },
            "session_summary": self._summary_generator.summary.to_dict(),
            "write_back_queue": {
                "pending": self.write_back_queue.pending_count(),
            },
        }

    # ── Auto-Compact (Phase 2.5) ──

    def should_compact(self) -> bool:
        """Check if dialogue compaction should be triggered."""
        if not self.config.compact_enabled:
            return False
        return self.working_memory.should_compact(self.config.compact_threshold)

    def check_and_compact(
        self,
        llm_client,
        model: str,
        task_brief: str = "",
        env_state: str = "",
    ):
        """
        Check and execute dialogue compaction + post-compact rehydration.

        Called by ChatAgentHandler after assistant response completes.
        task_brief / env_state used for re-injecting critical context after compaction.
        """
        if not self.should_compact():
            return

        if self.config.compact_save_transcript:
            transcript = self.working_memory.get_full_transcript()
            self.write_back_queue.enqueue(WriteBackItem(
                item_type="transcript",
                content=transcript,
                importance=0.3,
                metadata={"reason": "auto_compact"},
            ))

        summary = self.working_memory.compact(
            llm_client=llm_client,
            model=model,
            keep_recent=self.config.compact_keep_recent,
        )
        if summary:
            logger.info(
                f"[SessionMemoryManager] auto-compact done: "
                f"summary={len(summary)} chars"
            )
            self._rehydrate(task_brief, env_state)

    def _rehydrate(self, task_brief: str, env_state: str):
        """Post-compact rehydration: re-inject critical context that might be lost."""
        from handlers.agent.memory.working_memory import DialogueTurn

        parts = []
        if self.config.rehydrate_task_brief and task_brief:
            parts.append(f"[Current Active Task]\n{task_brief}")
        if self.config.rehydrate_env_state and env_state:
            parts.append(f"[Current Environment State]\n{env_state}")
        if parts:
            rehydrated = "\n\n".join(parts)
            self.working_memory.add_turn(DialogueTurn(
                role="system",
                content=f"<rehydrated-context>\n{rehydrated}\n</rehydrated-context>",
                trigger_type="system",
            ))
            logger.info(
                f"[SessionMemoryManager] rehydrated {len(parts)} context sections "
                f"({len(rehydrated)} chars)"
            )

    # ── Lifecycle ──

    def _maybe_update_summary(self):
        if self._summary_generator.should_update(self.working_memory.turn_count):
            self._summary_generator.update(self.working_memory)

    def flush_write_back(self) -> int:
        """Manually trigger write-back queue flush."""
        return self.write_back_queue.flush()

    def destroy(self):
        """Cleanup upon session termination."""
        self.write_back_queue.shutdown()
        self.working_memory.clear()
        self.perception_buffer.clear()
        self._summary_generator.clear()
        logger.info("[SessionMemoryManager] destroyed")

