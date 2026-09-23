"""
PromptCompiler — 4-Layer Prompt Orchestrator

Persona master profiles reside in OC; real-time system/context final assembly occurs in OAC.
OAC reconstructs the prompt every turn according to 4 layers below, rather than copying OC's system prompt directly.

4-Layer Architecture:
  L1 Stable Core         — OAC's own real-time rules, safety boundaries, output constraints + tool usage guidance
  L2 Persona Snapshot    — Trimmed snapshot compiled from OC persona master profile
  L3 Environment State   — Persistent environment state snapshot, wrapped in <environment-state>, injected at end
  L4 Recent Dialogue     — Discrete event <observation> + dialogue history + current user message

L1-L3 concatenated into system message (L3 near bottom); L4 concatenated into messages list.

Original L3(Mode), L4(TaskBrief), L5(RetrievedMemory) have been removed (Phase 4.2),
with corresponding info retrieved by LLM on demand via tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

# ── Layer Constants ──

LAYER_STABLE_CORE = "stable_core"
LAYER_PERSONA_SNAPSHOT = "persona_snapshot"
LAYER_ENVIRONMENT_STATE = "environment_state"
LAYER_RECENT_DIALOGUE = "recent_dialogue"

ALL_LAYERS = [
    LAYER_STABLE_CORE,
    LAYER_PERSONA_SNAPSHOT,
    LAYER_ENVIRONMENT_STATE,
    LAYER_RECENT_DIALOGUE,
]


# ── Data Models ──

@dataclass
class PromptInput:
    """Context passed from ChatAgent internally to PromptCompiler, not serialized across handlers."""
    user_message: str = ""
    trigger_type: str = "user"          # user / event
    response_hint: str = ""             # Free text description when event triggers
    persona_snapshot: str = ""          # Empty = use Compiler local default
    environment_state: str = ""         # L3 persistent environment state
    perception_events: List[Dict] = field(default_factory=list)
    dialogue_history: List[Dict] = field(default_factory=list)


@dataclass
class PromptLayerConfig:
    """Single layer configuration."""
    enabled: bool = True
    max_chars: int = 0          # 0 = unlimited
    section_header: str = ""    # Section header tag, e.g. "[Persona]"


@dataclass
class CompiledPrompt:
    """PromptCompiler output, directly sendable to LLM."""
    system_message: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)

    @property
    def full_messages(self) -> List[Dict[str, str]]:
        result = []
        if self.system_message:
            result.append({"role": "system", "content": self.system_message})
        result.extend(self.messages)
        return result

    @property
    def message_count(self) -> int:
        return len(self.messages) + (1 if self.system_message else 0)

    def __repr__(self) -> str:
        return (
            f"CompiledPrompt(system={len(self.system_message)} chars, "
            f"messages={len(self.messages)})"
        )


# ── Default Layer Configurations ──

DEFAULT_LAYER_CONFIGS: Dict[str, PromptLayerConfig] = {
    LAYER_STABLE_CORE: PromptLayerConfig(),
    LAYER_PERSONA_SNAPSHOT: PromptLayerConfig(
        section_header="[Persona]",
    ),
    LAYER_ENVIRONMENT_STATE: PromptLayerConfig(
        max_chars=800,
    ),
    LAYER_RECENT_DIALOGUE: PromptLayerConfig(),
}


# ── Default Texts ──

DEFAULT_STABLE_CORE = """\
You are a real-time digital avatar assistant equipped with vision capabilities, engaging in face-to-face conversation with the user through a camera.

## Output Constraints
- Answer in conversational, natural language with a warm, friendly tone
- Keep responses short (typically 2-3 sentences), unless the user explicitly requests a detailed explanation
- Do NOT output Markdown formatting, code blocks, or bullet point lists
- Do NOT refer to yourself as an "AI" or "language model"; speak authentically as the avatar persona itself

## Vision Rules
- The <environment-state> tag describes the persistent environmental state currently visible to you
- You may naturally mention observed content, but do not mention it in every sentence
- When the user asks "Can you see me?" and an environmental description is available, answer affirmatively and describe what you see

## Tool Usage
You possess the following capabilities; use them proactively at appropriate moments:
- When the user mentions past events, preferences, or prior agreements → use memory_search to query long-term memory
- When the user asks "What tasks were assigned to you?", or asks for reminders/schedules → use list_scheduled_tasks to check
- When uncertain about user address or your own persona settings → use get_agent_profile to fetch details
- When the user's question requires real-time data (e.g. time, weather) → use the corresponding tool
- Certain tools operate asynchronously; after invocation you will receive a {"status": "submitted"} state, decide your response accordingly
- Once a tool returns results, blend the outcome naturally into spoken text without reciting JSON verbatim

## Structured Tag Descriptions
You will receive the following structured tags in conversation:
- <environment-state>: Description of the ongoing environment state, to be understood as background knowledge rather than newly occurred events.
- <observation event="true">: An event that just occurred in the environment; consider whether a response is required.
- <observation compact="true">: Historical perception summary, for reference only.
- <background-results>: Background task completion notification; decide autonomously whether to inform the user.
- <dialogue-summary>: Compact summary of prior conversation, helping you maintain context continuity.
- <reminder>: System reminder of unresolved items requiring confirmation; please remind the user to address them.

## Approval Workflow
When background task notifications include an approval request (exec approval):
1. Record the pending confirmation item using the pending_confirmations tool (status=pending, source=exec_approval)
2. Inform the user of the approval details in natural language and ask for consent
3. After the user responds, submit the approval decision using the exec_approve tool
4. Update item status to confirmed/denied using the pending_confirmations tool"""

DEFAULT_PERSONA_SNAPSHOT = """\
Personality: Friendly, warm, communicates like a close friend. Observant, notices subtle changes in user state.
Tone: Relaxed, natural, showing genuine care for user emotions."""

MANDATORY_DELEGATION_POLICY = """\
## OAC-OC Collaboration Boundary (Strict Constraint)
Your role in the system is "Front Desk / Concierge". Actual functional task execution is handled by the OpenClaw (OC) backend.

- For any "functional tasks" or "complex tasks", you MUST invoke tools first; never claim verbally that a task has been completed before tools run.
- Examples of functional tasks: setting/modifying/cancelling reminders and schedules, creating/executing tasks, multi-step operations, background tracking affairs.
- When a task requires OC execution, prioritize calling spawn_agent, specifying subagent_type="oc_delegate".
- Before receiving tool return values, do NOT state "Completed". If a tool returns submitted_async, state clearly "Submitted to OpenClaw backend for processing".

### Background Approval Handling
- When an approval request appears in <background-results> or <pending-approvals>, it indicates the OC backend is waiting for approval.
- Step 1: Explain in natural spoken language what command the backend needs to execute, and inform the user of three options: allow once, allow always, or deny.
- Step 2: Call exec_approve based on user response, with decision as "allow-once" (this time only), "allow-always" (always allow similar commands), or "deny" (reject).
- When approval pops up repeatedly for the same task, proactively suggest the user choose "allow always" to minimize disruption.
- CRITICAL: Verbal phrases like "Approved" or "Confirmed" have NO effect! Only calling the exec_approve tool truly grants approval. Without calling the tool, backend approvals will remain blocked.
- Never ignore approval notifications, and never make decisions on behalf of the user arbitrarily.
"""

REALTIME_AND_TRUTH_POLICY = """\
## Truthfulness and Real-time Information (Strict Constraint)
- Real-time information such as current time, date, day of week, weather, and system status MUST rely on tool results as the single source of truth.
- For questions like "What time is it?" or "What day is today?", call corresponding tools first before answering.
- Do NOT guess the current time based on historical dialogue, prior responses, contextual association, or model memory.
- The same applies to task status: only state "Created", "Set", or "In Effect" after receiving tool results. Otherwise, state "I will submit for processing" or "Currently verifying".
"""


# ── PromptCompiler ──

class PromptCompiler:
    """
    4-Layer Prompt Orchestrator.

    Receives PromptInput (constructed inside ChatAgent) and outputs CompiledPrompt (sent directly to LLM).

    Usage::

        compiler = PromptCompiler()
        compiled = compiler.compile(prompt_input)
        messages = compiled.full_messages
    """

    def __init__(
        self,
        stable_core: str = DEFAULT_STABLE_CORE,
        persona_snapshot: str = DEFAULT_PERSONA_SNAPSHOT,
        layer_configs: Optional[Dict[str, PromptLayerConfig]] = None,
        max_dialogue_turns: Optional[int] = None,
    ):
        self.stable_core = stable_core
        self._persona_snapshot = persona_snapshot
        self.layer_configs = {**DEFAULT_LAYER_CONFIGS, **(layer_configs or {})}
        self.max_dialogue_turns = max_dialogue_turns

    # ── Dynamic Persona Snapshot Updates ──

    def update_persona_snapshot(self, snapshot: str):
        """Called when OC pushes a new persona snapshot."""
        self._persona_snapshot = snapshot
        logger.info(f"[PromptCompiler] persona snapshot updated ({len(snapshot)} chars)")

    @property
    def persona_snapshot(self) -> str:
        return self._persona_snapshot

    # ── Core Orchestration ──

    def compile(self, pi: PromptInput) -> CompiledPrompt:
        """
        Orchestrate PromptInput into CompiledPrompt.

        L1-L3 → system_message (L3 Environment State injected near bottom)
        L4    → messages list (discrete event <observation> + dialogue history + current turn)
        """
        system_parts: List[str] = []

        # L1 Stable Core (includes tool guidance + tag interpretation directives)
        self._append_layer(system_parts, LAYER_STABLE_CORE, self.stable_core)

        # L2 Persona Snapshot — Prioritize OC-provided snapshot in pi, fallback to local default
        snapshot = pi.persona_snapshot or self._persona_snapshot
        self._append_layer(system_parts, LAYER_PERSONA_SNAPSHOT, snapshot)

        # L3 Environment State — Wrapped in <environment-state>, injected at bottom
        env_state = self._build_environment_state(pi)
        self._append_layer(system_parts, LAYER_ENVIRONMENT_STATE, env_state)

        system_message = "\n\n".join(system_parts)

        # L4 Recent Dialogue + Discrete event <observation> + CurrentTurn
        messages = self._build_messages(pi)

        compiled = CompiledPrompt(
            system_message=system_message,
            messages=messages,
        )

        logger.debug(
            f"[PromptCompiler] compiled: system={len(system_message)} chars, "
            f"messages={len(messages)}, layers={len(system_parts)}"
        )
        return compiled

    # ── Layer Builders ──

    def _build_environment_state(self, pi: PromptInput) -> str:
        """L3: Wrap persistent environment state in <environment-state> tag."""
        if not pi.environment_state:
            return ""
        return (
            f"<environment-state>\n"
            f"{pi.environment_state}\n"
            f"</environment-state>"
        )

    def _build_messages(self, pi: PromptInput) -> List[Dict[str, str]]:
        """L4: Dialogue history + discrete events + current turn."""
        cfg = self.layer_configs.get(LAYER_RECENT_DIALOGUE)
        if cfg and not cfg.enabled:
            return []

        messages: List[Dict[str, str]] = []

        # Dialogue History
        history = pi.dialogue_history
        if self.max_dialogue_turns and len(history) > self.max_dialogue_turns:
            history = history[-self.max_dialogue_turns:]

        messages.extend(
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in history
            if m.get("content")
        )

        # Discrete events → <observation event="true"> user messages
        for event in pi.perception_events:
            obs_msg = self._format_observation(event)
            if obs_msg:
                messages.append({"role": "user", "content": obs_msg})

        # Current turn
        if pi.trigger_type == "user" and pi.user_message:
            messages.append({"role": "user", "content": pi.user_message})
        elif pi.trigger_type == "event":
            event_desc = pi.response_hint or "Environmental changes occurred; respond naturally according to context."
            messages.append({
                "role": "user",
                "content": (
                    f'<observation source="system" event="true">\n'
                    f'{event_desc}\n'
                    f'</observation>'
                ),
            })

        return messages

    @staticmethod
    def _format_observation(event: Dict) -> str:
        """Format a single discrete event into an <observation> tag."""
        content = event.get("content", "")
        if not content:
            return ""
        source = event.get("source", "camera")
        t = event.get("time", "")
        age = event.get("age_seconds", 0)
        age_str = f"{int(age)}s" if age else ""

        attrs = f'source="{source}"'
        if t:
            attrs += f' time="{t}"'
        if age_str:
            attrs += f' age="{age_str}"'
        attrs += ' event="true"'

        return f"<observation {attrs}>\n{content}\n</observation>"

    # ── Utility Methods ──

    def _append_layer(self, parts: List[str], layer_name: str, content: str):
        """Append a single layer of content according to configuration."""
        if not content:
            return

        cfg = self.layer_configs.get(layer_name)
        if cfg and not cfg.enabled:
            return

        if cfg and cfg.max_chars and len(content) > cfg.max_chars:
            content = content[:cfg.max_chars] + "…"

        if cfg and cfg.section_header:
            content = f"{cfg.section_header}\n{content}"

        parts.append(content)

