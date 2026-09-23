"""
OAC Prompt Compiler — 4-Layer Prompt Orchestration

Persona master profiles reside in OC; real-time system/context final assembly occurs in OAC.

Layer Architecture (Phase 4.2):
  L1  Stable Core         OAC's own real-time rules, safety boundaries, output constraints + tool usage guidance
  L2  Persona Snapshot    Trimmed snapshot compiled from OC persona master profile
  L3  Environment State   Persistent environment state snapshot (wrapped in <environment-state>, injected at end of system prompt)
  L4  Recent Dialogue     Discrete event <observation> + dialogue history + current user message

Original L3(Mode), L4(TaskBrief), L5(RetrievedMemory) have been removed,
with corresponding info retrieved by LLM on demand via tool calls.
"""
from handlers.agent.prompt.prompt_compiler import (
    PromptCompiler,
    PromptInput,
    PromptLayerConfig,
    CompiledPrompt,
    LAYER_STABLE_CORE,
    LAYER_PERSONA_SNAPSHOT,
    LAYER_ENVIRONMENT_STATE,
    LAYER_RECENT_DIALOGUE,
    ALL_LAYERS,
)

__all__ = [
    "PromptCompiler",
    "PromptInput",
    "PromptLayerConfig",
    "CompiledPrompt",
    "LAYER_STABLE_CORE",
    "LAYER_PERSONA_SNAPSHOT",
    "LAYER_ENVIRONMENT_STATE",
    "LAYER_RECENT_DIALOGUE",
    "ALL_LAYERS",
]
