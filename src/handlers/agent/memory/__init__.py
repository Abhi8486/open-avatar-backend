"""
OAC Memory Layer

Hierarchical memory system providing unified memory management for the Multi-Agent pipeline.

Layer Hierarchy:
- WorkingMemory: Session-level working memory (recent dialogue, current intent, session mode)
- PerceptionBuffer: Perception event buffer (deduplication, aggregation, TTL, importance rating)
- SessionSummary: Session summary (compressed representation of current session)
- WriteBackQueue: Asynchronous write-back queue (writing to external long-term memory system)
- SessionMemoryManager: Unified management layer coordinating all above components
"""

from handlers.agent.memory.working_memory import WorkingMemory, DialogueTurn
from handlers.agent.memory.perception_buffer import PerceptionBuffer, PerceptionEntry
from handlers.agent.memory.session_summary import SessionSummary
from handlers.agent.memory.write_behind_queue import WriteBackQueue, WriteBackItem, LocalWriteBackQueue
from handlers.agent.memory.session_memory_manager import SessionMemoryManager

__all__ = [
    "WorkingMemory",
    "DialogueTurn",
    "PerceptionBuffer",
    "PerceptionEntry",
    "SessionSummary",
    "WriteBackQueue",
    "WriteBackItem",
    "LocalWriteBackQueue",
    "SessionMemoryManager",
]
