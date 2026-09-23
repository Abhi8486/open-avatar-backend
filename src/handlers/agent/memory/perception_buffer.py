"""
Perception Buffer — Perception Event Buffer Pool

Features:
- Time decay + Hard TTL expiration
- Aggregation of similar events (merging counts of repeated similar events in short time windows)
- Importance rating and write-back filtering interface
- Capacity limits and automatic eviction

Design Principles:
- Perception events are not naturally long-term memory; most expire and are discarded
- Only high importance events are worth writing to OC via WriteBackQueue
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

from loguru import logger


@dataclass
class PerceptionEntry:
    """Perception buffer entry."""
    content: str
    category: str                       # scene / event / user_action / conversation
    importance: float                   # Initial importance (0-1)
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

    effective_importance: float = 0.0   # Effective importance after decay
    ttl: float = 0.0                    # Absolute expiration timestamp (0 = no expiration)
    occurrence_count: int = 1           # Aggregated count

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl <= 0:
            return False
        return (now or time.time()) >= self.ttl

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "effective_importance": self.effective_importance,
            "occurrence_count": self.occurrence_count,
        }


# Default TTL mappings (seconds)
DEFAULT_CATEGORY_TTL = {
    "scene": 300.0,         # Scene descriptions expire after 5 minutes
    "event": 120.0,         # Interaction events expire after 2 minutes
    "user_action": 180.0,   # User actions expire after 3 minutes
    "conversation": 600.0,  # Conversation snippets expire after 10 minutes
}

# Default importance threshold: events exceeding this threshold are worthy of writing back to OC
WRITEBACK_IMPORTANCE_THRESHOLD = 0.6


class PerceptionBuffer:
    """
    Perception event buffer pool.

    - Time decay + hard TTL expiration
    - Aggregation of similar events (same event_type + category within aggregation window)
    - Importance rating and write-back filtering interface
    - Capacity limits (evicting entries with lowest effective importance)
    """

    def __init__(
        self,
        max_entries: int = 100,
        decay_rate: float = 0.1,
        decay_interval: float = 60.0,
        category_ttl: Optional[Dict[str, float]] = None,
        aggregation_window: float = 10.0,
    ):
        self.max_entries = max_entries
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.category_ttl = category_ttl or dict(DEFAULT_CATEGORY_TTL)
        self.aggregation_window = aggregation_window

        self.entries: List[PerceptionEntry] = []
        self.last_decay_time: float = time.time()

    # ── Write ──

    def add(
        self,
        content: str,
        category: str,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
        event_type: Optional[str] = None,
    ):
        """
        Add perception entry.

        If a similar event exists within the aggregation window (matching category + event_type),
        merge instead of adding a new entry.
        """
        now = time.time()

        if event_type and self._try_aggregate(category, event_type, content, now):
            return

        ttl_seconds = self.category_ttl.get(category, 0.0)
        entry = PerceptionEntry(
            content=content,
            category=category,
            importance=importance,
            timestamp=now,
            metadata=metadata or {},
            effective_importance=importance,
            ttl=now + ttl_seconds if ttl_seconds > 0 else 0.0,
        )
        if event_type:
            entry.metadata["event_type"] = event_type

        self.entries.append(entry)
        self._maybe_decay()
        self._cleanup_expired()
        self._enforce_capacity()

        logger.debug(f"[PerceptionBuffer] +entry [{category}] {content[:50]}...")

    def _try_aggregate(
        self, category: str, event_type: str, content: str, now: float
    ) -> bool:
        """Attempt to aggregate new event into existing similar entry."""
        for entry in reversed(self.entries):
            if entry.category != category:
                continue
            if entry.metadata.get("event_type") != event_type:
                continue
            if now - entry.timestamp > self.aggregation_window:
                continue
            entry.occurrence_count += 1
            entry.content = content
            entry.timestamp = now
            # Aggregation boosts effective importance (capped at 1.0)
            entry.effective_importance = min(
                1.0, entry.effective_importance + 0.05
            )
            logger.debug(
                f"[PerceptionBuffer] aggregated [{category}/{event_type}] "
                f"count={entry.occurrence_count}"
            )
            return True
        return False

    # ── Query ──

    def query(
        self,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[PerceptionEntry]:
        self._maybe_decay()
        self._cleanup_expired()

        filtered = self.entries
        if category:
            filtered = [e for e in self.entries if e.category == category]

        sorted_entries = sorted(
            filtered, key=lambda x: x.effective_importance, reverse=True
        )
        return sorted_entries[:top_k]

    def get_summary(
        self,
        max_length: int = 500,
        categories: Optional[List[str]] = None,
    ) -> str:
        self._maybe_decay()
        self._cleanup_expired()

        filtered = self.entries
        if categories:
            filtered = [e for e in self.entries if e.category in categories]

        sorted_entries = sorted(
            filtered, key=lambda x: x.effective_importance, reverse=True
        )

        parts = []
        total_length = 0
        for entry in sorted_entries:
            count_tag = f"(x{entry.occurrence_count})" if entry.occurrence_count > 1 else ""
            entry_text = f"[{entry.category}]{count_tag} {entry.content}"
            if total_length + len(entry_text) > max_length:
                break
            parts.append(entry_text)
            total_length += len(entry_text) + 1
        return "\n".join(parts) if parts else ""

    def get_recent_summary(
        self, seconds: float = 60.0, max_length: int = 300
    ) -> str:
        cutoff = time.time() - seconds
        recent = [e for e in self.entries if e.timestamp >= cutoff and not e.is_expired()]
        sorted_entries = sorted(recent, key=lambda x: x.timestamp, reverse=True)

        parts = []
        total_length = 0
        for entry in sorted_entries:
            count_tag = f"(x{entry.occurrence_count})" if entry.occurrence_count > 1 else ""
            entry_text = f"[{entry.category}]{count_tag} {entry.content}"
            if total_length + len(entry_text) > max_length:
                break
            parts.append(entry_text)
            total_length += len(entry_text) + 1
        return "\n".join(parts) if parts else ""

    # ── Write-Back Filtering ──

    def get_writeback_candidates(
        self,
        threshold: float = WRITEBACK_IMPORTANCE_THRESHOLD,
    ) -> List[PerceptionEntry]:
        """Return entries with importance exceeding threshold, suitable for writing to OC long-term memory."""
        return [
            e for e in self.entries
            if e.effective_importance >= threshold and not e.is_expired()
        ]

    # ── Decay & Cleanup ──

    def _maybe_decay(self):
        now = time.time()
        if now - self.last_decay_time < self.decay_interval:
            return
        elapsed = now - self.last_decay_time
        decay_cycles = int(elapsed / self.decay_interval)
        if decay_cycles > 0:
            decay_factor = (1 - self.decay_rate) ** decay_cycles
            for entry in self.entries:
                entry.effective_importance *= decay_factor
            self.last_decay_time = now
            logger.debug(
                f"[PerceptionBuffer] decay: {decay_cycles} cycles, "
                f"factor={decay_factor:.3f}"
            )

    def _cleanup_expired(self):
        now = time.time()
        before = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired(now)]
        removed = before - len(self.entries)
        if removed:
            logger.debug(f"[PerceptionBuffer] expired {removed} entries")

    def _enforce_capacity(self):
        if len(self.entries) <= self.max_entries:
            return
        self.entries.sort(key=lambda x: x.effective_importance, reverse=True)
        removed = self.entries[self.max_entries:]
        self.entries = self.entries[: self.max_entries]
        if removed:
            logger.debug(f"[PerceptionBuffer] capacity: removed {len(removed)} entries")

    # ── Layered L6 / L7 Outputs (Phase 2.5) ──

    def get_state_summary(self, max_length: int = 500) -> str:
        """
        L6: Persistent environment state snapshot (latest scene complete + older scenes micro-compacted).
        Used for injection into system prompt's <environment-state> tag.
        """
        self._maybe_decay()
        self._cleanup_expired()

        now = time.time()
        scene_entries = [
            e for e in self.entries
            if e.category == "scene" and not e.is_expired(now)
        ]
        scene_entries.sort(key=lambda x: x.timestamp, reverse=True)

        if not scene_entries:
            return ""

        parts: List[str] = []
        total_len = 0

        latest = scene_entries[0]
        parts.append(latest.content)
        total_len += len(latest.content)

        compact_parts: List[str] = []
        for entry in scene_entries[1:]:
            compact = self._micro_compact_entry(entry)
            if total_len + len(compact) + 1 > max_length:
                break
            compact_parts.append(compact)
            total_len += len(compact) + 1

        if compact_parts:
            parts.append(" ".join(compact_parts))

        return "\n".join(parts)

    def get_recent_events(
        self,
        max_age: float = 60.0,
        categories: Optional[List[str]] = None,
    ) -> List[PerceptionEntry]:
        """
        L7: Return recent discrete event entries, used for injecting into history's <observation> tag.
        """
        now = time.time()
        cutoff = now - max_age
        target_categories = categories or ["event", "user_action"]

        events = [
            e for e in self.entries
            if e.category in target_categories
            and e.timestamp >= cutoff
            and not e.is_expired(now)
        ]
        events.sort(key=lambda x: x.timestamp)
        return events

    def get_compact_history(self, max_full: int = 2, max_length: int = 500) -> str:
        """
        Return perception history in micro-compact format:
        Recent max_full entries keep full content, earlier ones are compressed into single-line placeholders.
        """
        self._maybe_decay()
        self._cleanup_expired()

        if not self.entries:
            return ""

        sorted_entries = sorted(self.entries, key=lambda x: x.timestamp, reverse=True)
        parts: List[str] = []
        total_len = 0

        for i, entry in enumerate(sorted_entries):
            if i < max_full:
                text = entry.content
            else:
                text = self._micro_compact_entry(entry)

            if total_len + len(text) + 1 > max_length:
                break
            parts.append(text)
            total_len += len(text) + 1

        parts.reverse()
        return "\n".join(parts)

    @staticmethod
    def _micro_compact_entry(entry: "PerceptionEntry") -> str:
        """Compress a single perception entry into a single-line placeholder. Pure rule-driven, no LLM consumed."""
        icons = {"scene": "📷", "event": "⚡", "user_action": "👤", "conversation": "💬"}
        icon = icons.get(entry.category, "📌")
        t = time.strftime("%H:%M", time.localtime(entry.timestamp))
        summary = entry.content[:30].replace("\n", " ").strip()
        if len(entry.content) > 30:
            summary += "…"
        count = f"(x{entry.occurrence_count})" if entry.occurrence_count > 1 else ""
        return f"[{icon} {t}{count} {summary}]"

    # ── Tools ──

    def clear(self):
        self.entries.clear()
        logger.info("[PerceptionBuffer] cleared")

    @property
    def size(self) -> int:
        return len(self.entries)

    def get_stats(self) -> Dict:
        categories: Dict[str, int] = {}
        for entry in self.entries:
            categories[entry.category] = categories.get(entry.category, 0) + 1
        return {
            "total_entries": len(self.entries),
            "max_entries": self.max_entries,
            "categories": categories,
            "last_decay_time": self.last_decay_time,
        }

