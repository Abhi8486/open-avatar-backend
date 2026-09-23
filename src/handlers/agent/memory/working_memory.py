"""
Working Memory — Session-level Working Memory

Maintains immediate state of current session: recent dialogue turns, current intent, session mode,
latest task summary (Phase 3 filled by OC), pending confirmation items.

Phase 2.5 additions:
- auto-compact: LLM compression triggered when dialogue turn count exceeds threshold; old turns replaced with summary
- Save full transcript to WriteBackQueue before compression

Design Principles:
- Preserve only short-term info directly relevant to current interaction
- Discard all content upon session end
- Provide structured output oriented for Prompt compiler
"""
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from loguru import logger


@dataclass
class DialogueTurn:
    """Unified dialogue turn model."""
    role: str                               # "user" / "assistant" / "system"
    content: str                            # Message text
    timestamp: float = 0.0

    intent: Optional[str] = None            # User intent (user turns only)
    trigger_type: str = "user"              # "user" / "event"
    perception_snapshot: Optional[str] = None  # Visual perception snapshot at turn moment

    def to_llm_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "trigger_type": self.trigger_type,
            "perception_snapshot": self.perception_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DialogueTurn":
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", 0.0),
            intent=data.get("intent"),
            trigger_type=data.get("trigger_type", "user"),
            perception_snapshot=data.get("perception_snapshot"),
        )


class WorkingMemory:
    """
    Session-level working memory.

    Manages:
    - recent_dialogue: Recent N dialogue turns
    - current_intent: Currently recognized user intent
    - session_mode: Current session mode (chitchat / collaboration / report / ...)
    - last_task_summary: Latest task summary (Phase 3 filled by OC Bridge)
    - pending_confirmations: Pending user confirmation items
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._dialogue: List[DialogueTurn] = []

        self.current_intent: Optional[str] = None
        self.session_mode: str = "chitchat"
        self.last_task_summary: Optional[str] = None
        self.pending_confirmations: List[str] = []

        self._compact_summary: Optional[str] = None

    # ── Dialogue Management ──

    def add_turn(self, turn: DialogueTurn):
        self._dialogue.append(turn)
        self._enforce_limit()
        logger.debug(
            f"[WorkingMemory] +turn role={turn.role} "
            f"len={len(turn.content)} total={len(self._dialogue)}"
        )

    def add_user_turn(
        self,
        content: str,
        intent: Optional[str] = None,
        trigger_type: str = "user",
        perception_snapshot: Optional[str] = None,
    ):
        self.add_turn(DialogueTurn(
            role="user",
            content=content,
            timestamp=time.time(),
            intent=intent,
            trigger_type=trigger_type,
            perception_snapshot=perception_snapshot,
        ))
        if intent:
            self.current_intent = intent

    def add_assistant_turn(self, content: str):
        self.add_turn(DialogueTurn(
            role="assistant",
            content=content,
            timestamp=time.time(),
        ))

    def add_system_turn(self, content: str, trigger_type: str = "event"):
        self.add_turn(DialogueTurn(
            role="system",
            content=content,
            timestamp=time.time(),
            trigger_type=trigger_type,
        ))

    def get_recent_turns(self, n: Optional[int] = None) -> List[DialogueTurn]:
        count = n if n is not None else self.max_turns
        return list(self._dialogue[-count:])

    def get_llm_messages(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Return messages list suitable for direct passing to LLM.

        If compact_summary exists, inserts a summary pair with <dialogue-summary> tag at top,
        letting LLM understand previous dialogue context.
        """
        messages: List[Dict[str, str]] = []
        if self._compact_summary:
            messages.append({
                "role": "user",
                "content": f"<dialogue-summary>\n{self._compact_summary}\n</dialogue-summary>",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood, I have noted the previous dialogue context.",
            })
        messages.extend(t.to_llm_message() for t in self.get_recent_turns(n))
        return messages

    @property
    def turn_count(self) -> int:
        return len(self._dialogue)

    @property
    def last_user_message(self) -> Optional[str]:
        for turn in reversed(self._dialogue):
            if turn.role == "user":
                return turn.content
        return None

    # ── Session State ──

    def update_session_mode(self, mode: str):
        if mode != self.session_mode:
            logger.info(f"[WorkingMemory] session_mode: {self.session_mode} -> {mode}")
            self.session_mode = mode

    def set_task_summary(self, summary: str):
        self.last_task_summary = summary

    def add_pending_confirmation(self, item: str):
        self.pending_confirmations.append(item)

    def resolve_confirmation(self, item: str):
        if item in self.pending_confirmations:
            self.pending_confirmations.remove(item)

    # ── Context Export ──

    def get_context_snapshot(self) -> Dict:
        """Export complete snapshot of current working memory for Prompt orchestration or debugging."""
        return {
            "turn_count": self.turn_count,
            "current_intent": self.current_intent,
            "session_mode": self.session_mode,
            "last_task_summary": self.last_task_summary,
            "pending_confirmations": list(self.pending_confirmations),
            "recent_dialogue": [t.to_dict() for t in self.get_recent_turns(5)],
        }

    # ── Auto-Compact (Phase 2.5) ──

    def should_compact(self, threshold: int) -> bool:
        """Check if dialogue turns exceed compaction threshold."""
        return len(self._dialogue) >= threshold

    COMPACT_SYSTEM_PROMPT = (
        "You are a dialogue compression assistant. Analyze key dialogue points in <analysis> first, then output a concise summary in <summary>.\n\n"
        "The summary must retain the following 6 categories of information (skip missing categories):\n"
        "1. **User preferences and instructions** — Explicitly expressed likes, requirements, habits\n"
        "2. **Established facts in conversation** — Mutually confirmed information, names, numbers\n"
        "3. **Unfinished tasks and commitments** — Pending items, agreements, unresolved questions\n"
        "4. **Current emotion and interaction state** — User emotional state, interaction mode\n"
        "5. **Key user statements** — Concise quotes of critical statements\n"
        "6. **Environmental change summary** — Significant changes in sensor/visual observations\n\n"
        "Format:\n"
        "<analysis>\n(Reasoning and analysis process, omitted from final memory)\n</analysis>\n"
        "<summary>\n(Concise summary, within 300 words)\n</summary>"
    )

    def compact(
        self,
        llm_client,
        model: str,
        keep_recent: int = 5,
    ) -> Optional[str]:
        """
        Trigger auto-compact: LLM compresses old dialogue into summary using dual tags.

        Uses <analysis> + <summary> dual tag format, extracting <summary> content.
        Returns compressed summary text (for WriteBackQueue persistence), or None (if compaction unnecessary).
        """
        if len(self._dialogue) <= keep_recent:
            return None

        old_turns = self._dialogue[:-keep_recent]
        recent_turns = self._dialogue[-keep_recent:]

        old_text = "\n".join(
            f"[{t.role}] {t.content}" for t in old_turns
        )

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.COMPACT_SYSTEM_PROMPT},
                    {"role": "user", "content": old_text[:8000]},
                ],
                max_tokens=600,
            )
            raw_output = response.choices[0].message.content.strip()
            summary = self._extract_summary_tag(raw_output)
        except Exception as e:
            logger.warning(f"[WorkingMemory] compact LLM call failed: {e}")
            summary = f"(Previously conducted {len(old_turns)} dialogue turns)"

        if self._compact_summary:
            self._compact_summary = f"{self._compact_summary}\n{summary}"
        else:
            self._compact_summary = summary

        self._dialogue = recent_turns
        logger.info(
            f"[WorkingMemory] compacted: {len(old_turns)} turns → "
            f"{len(summary)} chars summary, kept {len(recent_turns)} recent"
        )
        return summary

    @staticmethod
    def _extract_summary_tag(text: str) -> str:
        """Extract content from <summary>...</summary> tags. Falls back to full text."""
        import re
        match = re.search(r"<summary>\s*(.*?)\s*</summary>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def get_full_transcript(self) -> str:
        """Export full dialogue transcript (including compact prefix) for WriteBackQueue storage."""
        parts = []
        if self._compact_summary:
            parts.append(f"[COMPACT SUMMARY]\n{self._compact_summary}")
        for t in self._dialogue:
            parts.append(json.dumps(t.to_dict(), ensure_ascii=False))
        return "\n".join(parts)

    # ── Internal ──

    def _enforce_limit(self):
        if len(self._dialogue) > self.max_turns:
            removed = len(self._dialogue) - self.max_turns
            self._dialogue = self._dialogue[-self.max_turns:]
            logger.debug(f"[WorkingMemory] trimmed {removed} old turns")

    def clear(self):
        self._dialogue.clear()
        self.current_intent = None
        self.session_mode = "chitchat"
        self.last_task_summary = None
        self.pending_confirmations.clear()
        self._compact_summary = None
        logger.info("[WorkingMemory] cleared")

