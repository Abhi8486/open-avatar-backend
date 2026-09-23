"""
Session Summary — Session Summary

Maintains compressed representation of current session for:
- Local fallback context when OC is unavailable
- Episodic event source for WriteBackQueue
- Session overview for Prompt Compiler

Phase 1 uses rule-based summary (keywords + intents + turn stats) without calling LLM.
Phase 2+ can connect to LLM for more precise compression.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from handlers.agent.memory.working_memory import WorkingMemory


@dataclass
class SessionSummary:
    """Session summary data."""
    summary_text: str = ""
    key_topics: List[str] = field(default_factory=list)
    key_intents: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_updated: float = 0.0

    def to_dict(self):
        return {
            "summary_text": self.summary_text,
            "key_topics": self.key_topics,
            "key_intents": self.key_intents,
            "turn_count": self.turn_count,
            "last_updated": self.last_updated,
        }


class SessionSummaryGenerator:
    """
    Session summary generator.

    Phase 1: Rule-based lightweight summary
    - Extract user intent list from recent dialogue
    - Count turns
    - Concatenate key phrases from recent N user messages

    Phase 2+: Replaceable with LLM-driven summary.
    """

    def __init__(
        self,
        update_interval_turns: int = 5,
        max_topics: int = 10,
    ):
        self.update_interval_turns = update_interval_turns
        self.max_topics = max_topics
        self._summary = SessionSummary()
        self._last_turn_count = 0

    @property
    def summary(self) -> SessionSummary:
        return self._summary

    def should_update(self, current_turn_count: int) -> bool:
        return (current_turn_count - self._last_turn_count) >= self.update_interval_turns

    def update(self, working_memory: "WorkingMemory"):
        """Update summary based on WorkingMemory's recent dialogue."""
        turns = working_memory.get_recent_turns()
        if not turns:
            return

        intents = []
        user_snippets = []
        for turn in turns:
            if turn.role == "user":
                if turn.intent:
                    intents.append(turn.intent)
                snippet = turn.content[:60].strip()
                if snippet:
                    user_snippets.append(snippet)

        unique_intents = list(dict.fromkeys(intents))[-self.max_topics:]
        recent_snippets = user_snippets[-self.max_topics:]

        summary_parts = []
        if unique_intents:
            summary_parts.append(f"User Intents: {', '.join(unique_intents)}")
        if recent_snippets:
            summary_parts.append(f"Recent Topics: {'; '.join(recent_snippets)}")
        summary_parts.append(f"Conducted {working_memory.turn_count} turns")

        if working_memory.session_mode != "chitchat":
            summary_parts.append(f"Current Mode: {working_memory.session_mode}")

        self._summary = SessionSummary(
            summary_text=" | ".join(summary_parts),
            key_topics=recent_snippets,
            key_intents=unique_intents,
            turn_count=working_memory.turn_count,
            last_updated=time.time(),
        )
        self._last_turn_count = working_memory.turn_count

        logger.debug(
            f"[SessionSummary] updated: {len(unique_intents)} intents, "
            f"{len(recent_snippets)} topics, {working_memory.turn_count} turns"
        )

    def force_update(self, working_memory: "WorkingMemory"):
        """Force update, ignoring interval."""
        self._last_turn_count = 0
        self.update(working_memory)

    def get_text(self) -> str:
        return self._summary.summary_text

    def clear(self):
        self._summary = SessionSummary()
        self._last_turn_count = 0
        logger.info("[SessionSummary] cleared")

