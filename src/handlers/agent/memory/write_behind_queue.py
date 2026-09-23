"""
Write-behind Queue — Asynchronous Write-Back Queue

Asynchronously writes information worthy of long-term preservation to external long-term memory system (Phase 3: OpenClaw),
without blocking the main real-time dialogue stream.

Phase 1: LocalWriteBackQueue — Logging only, no actual write to OC
Phase 3: McpWriteBackQueue — Write to OpenClaw via MCP
"""
import time
import threading
from abc import ABC, abstractmethod
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from loguru import logger


@dataclass
class WriteBackItem:
    """Write-back item."""
    item_type: str                      # "episodic" / "preference" / "fact" / "event"
    content: str
    importance: float = 0.5
    timestamp: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "item_type": self.item_type,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class WriteBackQueue(ABC):
    """Write-back queue abstract interface."""

    @abstractmethod
    def enqueue(self, item: WriteBackItem):
        """Enqueue item into write-back queue."""
        ...

    @abstractmethod
    def flush(self) -> int:
        """Immediately process all items in queue. Returns count processed."""
        ...

    @abstractmethod
    def pending_count(self) -> int:
        """Returns number of pending items in queue."""
        ...

    @abstractmethod
    def shutdown(self):
        """Shutdown queue and release resources."""
        ...


class LocalWriteBackQueue(WriteBackQueue):
    """
    Phase 1 implementation: Logging only.

    All enqueued items are written to logs only, not actually sent to external systems.
    Used for development debugging and Phase 1 behavior validation.
    """

    def __init__(self, max_queue_size: int = 1000):
        self._queue: deque[WriteBackItem] = deque(maxlen=max_queue_size)
        self._total_enqueued: int = 0
        self._total_flushed: int = 0
        self._lock = threading.Lock()

    def enqueue(self, item: WriteBackItem):
        with self._lock:
            self._queue.append(item)
            self._total_enqueued += 1
        logger.info(
            f"[WriteBackQueue:Local] enqueued [{item.item_type}] "
            f"importance={item.importance:.2f}: {item.content[:80]}..."
        )

    def flush(self) -> int:
        with self._lock:
            count = len(self._queue)
            if count == 0:
                return 0
            items = list(self._queue)
            self._queue.clear()
            self._total_flushed += count

        by_type = Counter(i.item_type for i in items)
        logger.info(
            f"[WriteBackQueue:Local] flushed {count} items "
            f"(by type: {dict(by_type)}; total flushed all-time: {self._total_flushed})"
        )
        # Per-item lines look like duplicates when many auto-compact snapshots
        # share the same [COMPACT SUMMARY] prefix (log truncates to 60 chars).
        for item in items:
            logger.debug(
                f"[WriteBackQueue:Local] flush item [{item.item_type}] "
                f"{item.content[:120]}..."
            )
        return count

    def pending_count(self) -> int:
        return len(self._queue)

    def shutdown(self):
        self.flush()

    def get_stats(self) -> Dict:
        return {
            "pending": len(self._queue),
            "total_enqueued": self._total_enqueued,
            "total_flushed": self._total_flushed,
        }

