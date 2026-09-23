"""
Chat Agent Handler — Unified Main Dialogue Agent

Merges all responsibilities of the former ManagerAgent (memory management, perception caching, proactive triggers, event handling)
and PersonaAgent (PromptCompiler orchestration, LLM invocation, streaming output).

Data Flow:
  HUMAN_TEXT / PERCEPTION_CONTEXT / ENVIRONMENT_EVENT → ChatAgent → AVATAR_TEXT

Perception serves as an independent asynchronous perception service (handler) pushing perception data and events to ChatAgent.
"""
import json
import os
import re
import threading
import time
from abc import ABC
from typing import Dict, List, Optional, Set, cast

from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.chat_stream_config import ChatStreamConfig
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

from handlers.agent.agent_data_models import PerceptionData, EnvironmentEvent
from handlers.agent.memory.session_memory_manager import SessionMemoryManager, MemoryConfig
from handlers.agent.tools.tool_registry import ToolRegistry
from handlers.agent.prompt.prompt_compiler import (
    PromptCompiler,
    PromptInput,
    PromptLayerConfig,
    DEFAULT_STABLE_CORE,
    DEFAULT_PERSONA_SNAPSHOT,
    MANDATORY_DELEGATION_POLICY,
    REALTIME_AND_TRUTH_POLICY,
    LAYER_PERSONA_SNAPSHOT,
    LAYER_ENVIRONMENT_STATE,
)


# ── Proactive Message Trigger Configurations ──

class EventTriggerConfig(BaseModel):
    """Trigger configuration for a single event type"""
    enabled: bool = True
    cooldown: float = 30.0
    hint: str = ""


class IdleTriggerConfig(BaseModel):
    """Idle trigger configuration"""
    enabled: bool = False
    idle_seconds: float = 60.0
    hint: str = "The user has been quiet for a while. You can check in on them or start a light conversation."
    mode_overrides: Dict[str, float] = Field(
        default={},
        description="Mode → Idle seconds override, e.g. {companion: 120, office: 300}"
    )


class PendingConfirmationTriggerConfig(BaseModel):
    """Pending confirmation item proactive reminder configuration"""
    enabled: bool = Field(default=True, description="Whether to enable pending confirmation proactive reminders")
    idle_seconds: float = Field(default=15.0, description="User idle seconds before triggering reminder")
    cooldown: float = Field(default=30.0, description="Cooldown time between reminders in seconds")
    hint: str = Field(
        default=(
            "There are pending approval requests. You must first explain the command details to the user and ask for consent. "
            "Upon receiving consent, immediately call the exec_approve tool. Verbal statements like 'Approved' are invalid."
        ),
        description="Hint text injected into the agent",
    )


class ProactiveConfig(BaseModel):
    """Master proactive message trigger configuration"""
    enabled: bool = Field(default=True, description="Master switch")
    event_triggers: Dict[str, EventTriggerConfig] = Field(
        default={
            "waving": EventTriggerConfig(
                hint="Please respond friendly to the user's greeting and ask how you can help."
            ),
            "showing_object": EventTriggerConfig(
                hint="The user is showing an object. Please inquire about or comment on the object.",
            ),
            "asking_for_attention": EventTriggerConfig(
                hint="The user is seeking your attention. Please respond actively and ask how to help."
            ),
            "arriving": EventTriggerConfig(
                cooldown=60.0,
                hint="Someone entered the frame. Welcome them appropriately.",
            ),
            "leaving": EventTriggerConfig(
                cooldown=60.0,
                hint="The user seems to be leaving. Say goodbye appropriately.",
            ),
        },
        description="Event type → Trigger configuration",
    )
    idle_trigger: IdleTriggerConfig = Field(
        default_factory=IdleTriggerConfig,
        description="Idle trigger configuration",
    )
    pending_confirmation_trigger: PendingConfirmationTriggerConfig = Field(
        default_factory=PendingConfirmationTriggerConfig,
        description="Pending confirmation item proactive reminder configuration",
    )


class ContextCompactConfig(BaseModel):
    """Dialogue context auto-compact configuration"""
    enabled: bool = Field(default=True, description="Whether auto-compact is enabled")
    compact_threshold: int = Field(default=15, description="Number of dialogue rounds that triggers compaction")
    keep_recent: int = Field(default=5, description="Number of recent turns retained after compaction")
    save_transcript: bool = Field(default=True, description="Whether to save complete transcript before compaction")
    compact_model: Optional[str] = Field(default=None, description="LLM model used for compaction (defaults to main model)")
    rehydrate_task_brief: bool = Field(default=True, description="Re-inject active task brief after compaction")
    rehydrate_env_state: bool = Field(default=True, description="Re-inject current environment state after compaction")


class OcBridgeConfig(BaseModel):
    """OpenClaw Bridge Configuration

    When enabled, activates:
    - Plugin Tools MCP (get_agent_profile, memory_search, etc.)
    - oac-bridge HTTP Channel (bidirectional messaging OAC ↔ OC)
    """
    enabled: bool = Field(default=False, description="Whether OC Bridge is enabled")
    plugin_tools_cmd: Optional[str] = Field(
        default=None,
        description="Plugin Tools MCP start command (e.g., 'node dist/mcp/plugin-tools-serve.js')"
    )
    persona_refresh_interval: float = Field(default=600.0, description="Persona snapshot refresh interval in seconds")
    task_mirror_path: str = Field(default=".oac_tasks/mirror.json", description="Task mirror JSON path")
    gateway_http_url: str = Field(
        default="http://localhost:18789",
        description="OC Gateway HTTP address (for sending webhooks)"
    )
    webhook_path: str = Field(
        default="/webhook/oac-bridge",
        description="oac-bridge webhook path on OC"
    )
    token: str = Field(
        default="",
        description="oac-bridge shared authentication token (empty means no auth)"
    )
    callback_port: int = Field(
        default=8011,
        description="OAC side callback HTTP service port (receives OC replies)"
    )


class ToolUseConfig(BaseModel):
    """Tool execution configuration"""
    enabled: bool = Field(default=True, description="Whether tool execution is enabled")
    max_tool_rounds: int = Field(default=5, description="Maximum tool execution rounds per handle call")
    register_demo_tools: bool = Field(default=True, description="Whether to register demo tools (for testing)")


class ChatAgentConfig(HandlerBaseConfigModel, BaseModel):
    """Chat Agent Configuration — Merges configuration of former Manager and Persona"""
    # LLM Config
    llm_model: str = Field(default="qwen-plus", description="LLM model name")
    api_key: Optional[str] = Field(default=None, description="API Key (defaults to environment variable)")
    api_url: Optional[str] = Field(default=None, description="API URL")
    enable_thinking: bool = Field(
        default=False,
        description="DashScope compatibility mode enable_thinking; injects extra_body when api_url points to dashscope",
    )

    # Tool execution
    tool_use: ToolUseConfig = Field(default_factory=ToolUseConfig, description="Tool execution config")

    # OpenClaw Bridge
    oc_bridge: OcBridgeConfig = Field(default_factory=OcBridgeConfig, description="OpenClaw Bridge config")

    # PromptCompiler L1 Stable Core
    stable_core: str = Field(default=DEFAULT_STABLE_CORE, description="OAC real-time rules / output constraints")

    # PromptCompiler L2 Persona Snapshot
    persona_snapshot: str = Field(
        default=DEFAULT_PERSONA_SNAPSHOT,
        description="Persona snapshot (local default, dynamically updated by OC later)"
    )

    # Layer character limits
    persona_max_chars: int = Field(default=2500, description="L2 Persona Snapshot max characters (IDENTITY+SOUL+USER)")
    perception_max_chars: int = Field(default=800, description="L3 Environment State max characters")

    # Dialogue history
    max_dialogue_turns: int = Field(default=20, description="Maximum dialogue history rounds")
    compiler_dialogue_turns: Optional[int] = Field(
        default=None,
        description="PromptCompiler side max dialogue turns (None = use all)"
    )

    # Memory config
    perception_max_entries: int = Field(default=100, description="Perception buffer pool size")
    perception_decay_rate: float = Field(default=0.1, description="Perception decay rate")
    perception_aggregation_window: float = Field(default=10.0, description="Perception event aggregation window in seconds")
    summary_update_interval: int = Field(default=5, description="Session summary update interval in turns")

    # Proactive message trigger
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig, description="Proactive message trigger config")

    # Dialogue context compaction
    context_compact: ContextCompactConfig = Field(
        default_factory=ContextCompactConfig, description="Dialogue context auto-compact config"
    )

    def to_memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            max_dialogue_turns=self.max_dialogue_turns,
            perception_max_entries=self.perception_max_entries,
            perception_decay_rate=self.perception_decay_rate,
            perception_aggregation_window=self.perception_aggregation_window,
            summary_update_interval_turns=self.summary_update_interval,
            compact_enabled=self.context_compact.enabled,
            compact_threshold=self.context_compact.compact_threshold,
            compact_keep_recent=self.context_compact.keep_recent,
            compact_save_transcript=self.context_compact.save_transcript,
            rehydrate_task_brief=self.context_compact.rehydrate_task_brief,
            rehydrate_env_state=self.context_compact.rehydrate_env_state,
        )


class ChatAgentContext(HandlerContext):
    """Chat Agent Context — Merges context of former Manager and Persona"""

    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[ChatAgentConfig] = None
        self.llm_client = None
        self.memory: Optional[SessionMemoryManager] = None
        self.compiler: Optional[PromptCompiler] = None
        self.tool_registry: Optional[ToolRegistry] = None

        # OC Bridge components
        self.oc_mcp_client = None
        self.oc_channel_client = None  # OcChannelClient (oac-bridge HTTP channel)
        self.persona_mgr = None       # PersonaSnapshotManager
        self.task_queue = None         # TaskNotificationQueue
        self.task_mirror = None        # TaskMirror — kept for compact rehydration
        self.pending_confirmations = None  # PendingConfirmationsManager

        # Perception cache
        self.cached_perception: Optional[PerceptionData] = None

        # Proactive triggers
        self.pending_events: List[EnvironmentEvent] = []
        self.output_definitions: Optional[Dict[ChatDataType, HandlerDataInfo]] = None

        # Input buffer
        self.input_buffer: str = ""
        self.is_generating: bool = False

        # Event response tracking
        self.responded_events: Dict[str, float] = {}

        # Idle trigger & proactive wake
        self.last_interaction_time: float = 0.0
        self._idle_stop: threading.Event = threading.Event()
        self._idle_triggered: bool = False
        self._proactive_wake: threading.Event = threading.Event()

        # Stream output management
        self.active_stream_keys: Set[str] = set()

        # Serialize _generate_response — the idle-trigger thread and the
        # pipeline thread must never call it concurrently.
        self._generate_lock: threading.Lock = threading.Lock()


class ChatAgentHandler(HandlerBase, ABC):
    """
    Chat Agent — Unified Main Dialogue Agent

    Responsibilities:
    1. Receive HUMAN_TEXT, PERCEPTION_CONTEXT, ENVIRONMENT_EVENT
    2. Maintain layered memory via SessionMemoryManager
    3. Orchestrate prompt via PromptCompiler 4-layer architecture (L1 Core + L2 Persona + L3 Env + L4 Dialogue)
    4. Call LLM with streaming + Agent Loop (tool_call → execute → feedback → repeat)
    5. Proactive message triggers (event-driven + idle-driven)
    """

    def __init__(self):
        super().__init__()
        self.output_definition: Optional[DataBundleDefinition] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=ChatAgentConfig)

    def load(self, engine_config: ChatEngineConfigModel,
             handler_config: Optional[HandlerBaseConfigModel] = None):
        self.output_definition = DataBundleDefinition()
        self.output_definition.add_entry(DataBundleEntry(name="avatar_text"))
        logger.info("ChatAgentHandler loaded")

    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        context = ChatAgentContext(session_context.session_info.session_id)

        if isinstance(handler_config, ChatAgentConfig):
            context.config = handler_config
        else:
            context.config = ChatAgentConfig()

        api_key = context.config.api_key or os.getenv("DASHSCOPE_API_KEY")
        try:
            from openai import OpenAI
            context.llm_client = OpenAI(api_key=api_key, base_url=context.config.api_url)
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")

        context.memory = SessionMemoryManager(config=context.config.to_memory_config())
        context.compiler = self._build_compiler(context.config)
        context.tool_registry = self._build_tool_registry(context.config)

        # OC Bridge initialization
        if context.config.oc_bridge.enabled:
            self._init_oc_bridge(context)
        else:
            logger.info(
                "[ChatAgent] OC Bridge not enabled: will not pull L2 persona from OpenClaw, "
                "nor register OC tools. Set ChatAgent.oc_bridge.enabled=true in config "
                "(and configure plugin_tools_cmd, gateway_http_url)."
            )

        context.last_interaction_time = time.time()

        logger.info(f"ChatAgentContext created for session {context.session_id}")
        return context

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(ChatAgentContext, handler_context)
        proactive_cfg = context.config.proactive
        need_loop = proactive_cfg.enabled and (
            proactive_cfg.idle_trigger.enabled
            or proactive_cfg.pending_confirmation_trigger.enabled
        )
        if need_loop:
            t = threading.Thread(
                target=self._idle_trigger_loop,
                args=(context,),
                daemon=True,
                name=f"chat-idle-{context.session_id}",
            )
            t.start()
            logger.info(
                f"[ChatAgent] Proactive trigger loop started "
                f"(idle={proactive_cfg.idle_trigger.enabled}, "
                f"pending_confirm={proactive_cfg.pending_confirmation_trigger.enabled})"
            )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry(name="avatar_text"))

        return HandlerDetail(
            inputs={
                ChatDataType.HUMAN_TEXT: HandlerDataInfo(type=ChatDataType.HUMAN_TEXT),
                ChatDataType.PERCEPTION_CONTEXT: HandlerDataInfo(type=ChatDataType.PERCEPTION_CONTEXT),
                ChatDataType.ENVIRONMENT_EVENT: HandlerDataInfo(type=ChatDataType.ENVIRONMENT_EVENT),
            },
            outputs={
                ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                    type=ChatDataType.AVATAR_TEXT,
                    definition=definition,
                ),
            },
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, None),
            ],
        )

    # ── Signal Handling ──

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        context = cast(ChatAgentContext, context)

        if signal.type == ChatSignalType.STREAM_CANCEL and signal.related_stream:
            stream_key = signal.related_stream.stream_key_str
            if stream_key is not None and stream_key in context.active_stream_keys:
                context.active_stream_keys.discard(stream_key)
                logger.info(f"[ChatAgent] Removed stream {stream_key} from active set")
            return

        if signal.type == ChatSignalType.ENVIRONMENT_EVENT:
            event = EnvironmentEvent.from_dict(signal.signal_data or {})
            importance = self._event_importance(event)
            if context.memory:
                context.memory.record_perception(
                    content=event.description,
                    category="event",
                    importance=importance,
                    metadata=event.to_dict(),
                    event_type=event.event_type,
                )
            context.pending_events.append(event)

    # ── Main Entrypoint ──

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(ChatAgentContext, context)

        logger.debug(
            f"[ChatAgent] Received input: type={inputs.type.value}, is_last={inputs.is_last_data}"
        )

        if inputs.type == ChatDataType.PERCEPTION_CONTEXT:
            self._handle_perception_context(context, inputs)
            return

        if inputs.type == ChatDataType.ENVIRONMENT_EVENT:
            self._handle_environment_event(context, inputs, output_definitions)
            return

        if inputs.type == ChatDataType.HUMAN_TEXT:
            self._handle_human_text(context, inputs, output_definitions)
            return

    # ── PERCEPTION_CONTEXT ──

    def _handle_perception_context(self, context: ChatAgentContext, inputs: ChatData):
        data = inputs.data.get_main_data()
        if data is None:
            return

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse perception data as JSON: {data[:100]}")
                return
        if isinstance(data, dict):
            context.cached_perception = PerceptionData.from_dict(data)
        elif isinstance(data, PerceptionData):
            context.cached_perception = data

        if context.memory and context.cached_perception:
            context.memory.record_perception(
                content=context.cached_perception.scene_summary,
                category="scene",
                importance=0.3,
            )

    # ── ENVIRONMENT_EVENT ──

    def _handle_environment_event(
        self, context: ChatAgentContext, inputs: ChatData,
        output_definitions: Dict[ChatDataType, HandlerDataInfo],
    ):
        data = inputs.data.get_main_data()
        if data is None:
            return

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return

        if not context.config.proactive.enabled:
            return

        event = EnvironmentEvent.from_dict(data)
        trigger_cfg = context.config.proactive.event_triggers.get(event.event_type)

        if trigger_cfg is None or not trigger_cfg.enabled:
            logger.debug(f"[ChatAgent] Event type not configured or disabled: {event.event_type}")
            return

        if self._should_skip_event(context, event, trigger_cfg.cooldown):
            logger.info(f"[ChatAgent] Skip already responded event: {event.event_type} (in cooldown)")
            return

        logger.info(
            f"[ChatAgent] Processing interaction event: {event.event_type} "
            f"(confidence: {event.confidence:.2f}, urgency: {event.urgency})"
        )

        importance = self._event_importance(event)
        if context.memory:
            context.memory.record_perception(
                content=event.description,
                category="event",
                importance=importance,
                metadata=event.to_dict(),
                event_type=event.event_type,
            )

        context.output_definitions = output_definitions

        if event.should_interrupt():
            self._handle_proactive_response(
                context, [event], output_definitions, [trigger_cfg]
            )
        elif event.should_respond_immediately():
            if not context.is_generating:
                self._handle_proactive_response(
                    context, [event], output_definitions, [trigger_cfg]
                )
            else:
                context.pending_events.append(event)
        else:
            context.pending_events.append(event)

    # ── Proactive Response ──

    def _handle_proactive_response(
        self,
        context: ChatAgentContext,
        events: List[EnvironmentEvent],
        output_definitions: Dict[ChatDataType, HandlerDataInfo],
        trigger_cfgs: Optional[List[Optional[EventTriggerConfig]]] = None,
    ):
        """Produce a consolidated response for a group of events."""
        if not events:
            return

        acquired = context._generate_lock.acquire(timeout=0.5)
        if not acquired:
            logger.info("[ChatAgent] Proactive response skipped: _generate_lock is busy")
            return
        try:
            context.is_generating = True

            cfgs = trigger_cfgs or [None] * len(events)

            hints = []
            for ev, cfg in zip(events, cfgs):
                if cfg and cfg.hint:
                    hints.append(cfg.hint)
                else:
                    hints.append(ev.description)

            merged_hint = "\n".join(hints)

            prompt_input = self._build_prompt_input(
                context,
                trigger_type="event",
                response_hint=merged_hint,
            )

            if context.memory:
                for ev in events:
                    context.memory.record_system_event(
                        content=f"[Environment Event: {ev.event_type}] {ev.description}",
                        trigger_type="event",
                    )

            event_names = ", ".join(ev.event_type for ev in events)
            logger.info(f"[ChatAgent] Proactive response ({len(events)} events merged): {event_names}")
            self._generate_response(context, prompt_input, output_definitions)

            now = time.time()
            for ev in events:
                context.responded_events[ev.event_type] = now
            context.last_interaction_time = now
            context._idle_triggered = False
        finally:
            context._generate_lock.release()

    # ── HUMAN_TEXT (Main user input workflow) ──

    def _handle_human_text(
        self, context: ChatAgentContext,
        inputs: ChatData,
        output_definitions: Dict[ChatDataType, HandlerDataInfo],
    ):
        if context.responded_events:
            context.responded_events.clear()
        if context.pending_events:
            logger.info(
                f"[ChatAgent] User input received, discarding {len(context.pending_events)} pending events"
            )
            context.pending_events.clear()

        text = inputs.data.get_main_data()
        if isinstance(text, str):
            context.input_buffer += text

        if not inputs.is_last_data:
            return

        full_text = context.input_buffer.strip()
        context.input_buffer = ""

        if not full_text:
            return

        # Wait for any in-flight proactive generation to finish
        acquired = context._generate_lock.acquire(timeout=10.0)
        if not acquired:
            logger.warning(
                f"[ChatAgent] User input '{full_text[:40]}' generation lock timeout, forcing continuation"
            )
        try:
            logger.info("[ChatAgent] ══════════════════════════════════════════")
            logger.info(f"[ChatAgent] Processing user input: '{full_text}'")
            context.is_generating = True
            context.last_interaction_time = time.time()
            context._idle_triggered = False
            context.output_definitions = output_definitions

            perception_snapshot = (
                context.cached_perception.scene_summary if context.cached_perception else None
            )
            if context.memory:
                context.memory.record_user_input(
                    content=full_text,
                    trigger_type="user",
                    perception_snapshot=perception_snapshot,
                )

            prompt_input = self._build_prompt_input(
                context,
                trigger_type="user",
            )

            self._generate_response(context, prompt_input, output_definitions)

            self._process_pending_events(context)
            logger.info("[ChatAgent] ══════════════════════════════════════════")
        finally:
            if acquired:
                context._generate_lock.release()

    # ── Build PromptInput ──

    def _build_prompt_input(
        self,
        context: ChatAgentContext,
        *,
        trigger_type: str = "user",
        response_hint: str = "",
    ) -> PromptInput:
        """Build PromptInput (4 layers) from internal agent state."""
        # L2: Persona snapshot — prefer OC, fallback to local
        persona_snapshot = ""
        if context.persona_mgr:
            persona_snapshot = context.persona_mgr.get_snapshot()

        # L3: Continuous environment state snapshot
        env_state = ""
        if context.memory:
            env_state = context.memory.get_environment_state(max_length=500)
        if not env_state and context.cached_perception:
            env_state = context.cached_perception.scene_summary

        # L4: Discrete events (from PerceptionBuffer, recorded via record_perception)
        perception_events = []
        if context.memory:
            perception_events = context.memory.get_recent_perception_events(max_age=60.0)

        # Drain OC background notifications and pending approvals into memory
        from handlers.agent.oc_bridge.oc_prompt_injection import (
            drain_task_notifications,
            inject_pending_approvals,
        )
        drain_task_notifications(context.task_queue, context.memory)
        inject_pending_approvals(context.pending_confirmations, context.memory)

        return PromptInput(
            trigger_type=trigger_type,
            response_hint=response_hint,
            persona_snapshot=persona_snapshot,
            environment_state=env_state,
            perception_events=perception_events,
            dialogue_history=(
                context.memory.get_dialogue_for_llm(context.config.max_dialogue_turns)
                if context.memory else []
            ),
        )

    # ── LLM Invocation + Streaming + Agent Loop ──

    def _generate_response(
        self,
        context: ChatAgentContext,
        prompt_input: PromptInput,
        output_definitions: Dict[ChatDataType, HandlerDataInfo],
    ):
        output_definition = output_definitions.get(ChatDataType.AVATAR_TEXT).definition
        streamer = context.data_submitter.get_streamer(ChatDataType.AVATAR_TEXT)

        if context.llm_client is None:
            logger.error("[ChatAgent] LLM client unavailable")
            output = DataBundle(output_definition)
            output.set_main_data("Sorry, I am temporarily unable to process your request. Please try again later.")
            streamer.stream_data(output, finish_stream=True)
            context.is_generating = False
            return

        stream_key = streamer.current_stream.identity.stream_key_str if streamer.current_stream is not None else None
        if stream_key is None:
            stream = streamer.new_stream(
                sources=[],
                config=ChatStreamConfig(cancelable=True),
            )
            stream_key = stream.stream_key_str

        compiled = context.compiler.compile(prompt_input)
        messages = compiled.full_messages

        logger.info(
            f"[ChatAgent] PromptCompiler: system={len(compiled.system_message)} chars, "
            f"messages={compiled.message_count}"
        )

        self._log_incomplete_work_summary(context)
        self._debug_log_prompt(context, compiled)

        if stream_key:
            context.active_stream_keys.add(stream_key)

        try:
            full_response = self._agent_loop(
                context, messages, output_definition, streamer, stream_key,
            )

            if full_response is None:
                context.is_generating = False
                return

            logger.info(f"[ChatAgent] Response: '{full_response[:80]}...'")

            if full_response and context.memory:
                context.memory.record_assistant_response(full_response)
                self._check_compact(context)

        except Exception as e:
            logger.error(f"[ChatAgent] LLM invocation failed: {e}")
            output = DataBundle(output_definition)
            output.set_main_data("Sorry, I am temporarily unable to process your request. Please try again later.")
            streamer.stream_data(output, finish_stream=True)
            context.is_generating = False
            return

        if stream_key:
            context.active_stream_keys.discard(stream_key)
        end_output = DataBundle(output_definition)
        end_output.set_main_data("")
        streamer.stream_data(end_output, finish_stream=True)

        context.is_generating = False
        context.last_interaction_time = time.time()

    @staticmethod
    def _apply_llm_extra_body(context: ChatAgentContext, kwargs: dict) -> None:
        """DashScope OpenAI compatibility interface requires enable_thinking passed via extra_body"""
        api_url = context.config.api_url
        if not api_url or "dashscope" not in api_url.lower():
            return
        extra = dict(kwargs.get("extra_body") or {})
        extra["enable_thinking"] = context.config.enable_thinking
        kwargs["extra_body"] = extra

    def _agent_loop(
        self,
        context: ChatAgentContext,
        messages: List[dict],
        output_definition,
        streamer,
        stream_key: Optional[str],
    ) -> Optional[str]:
        """Multi-step agent loop: LLM call → tool_use → feedback → repeat.

        Returns the final text response, or None if cancelled.
        """
        registry = context.tool_registry
        use_tools = (
            registry is not None
            and registry.has_tools()
            and context.config.tool_use.enabled
        )
        max_rounds = context.config.tool_use.max_tool_rounds

        for round_idx in range(max_rounds):
            tools_param = registry.get_schemas() if use_tools else None

            kwargs = dict(
                model=context.config.llm_model,
                messages=messages,
                stream=True,
            )
            if tools_param:
                kwargs["tools"] = tools_param

            self._apply_llm_extra_body(context, kwargs)

            response = context.llm_client.chat.completions.create(**kwargs)

            full_text, tool_calls, cancelled = self._stream_response(
                context, response, output_definition, streamer, stream_key,
            )

            if cancelled:
                logger.info("[ChatAgent] Stream cancelled during agent loop")
                return None

            if not tool_calls:
                return full_text

            # LLM requested tool calls — execute and continue the loop
            logger.info(
                f"[ChatAgent] Agent loop round {round_idx + 1}: "
                f"{len(tool_calls)} tool call(s)"
            )

            assistant_msg = {"role": "assistant", "content": full_text or None}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            interrupted_during_tools = False
            for tc in tool_calls:
                if stream_key and stream_key not in context.active_stream_keys:
                    logger.info(
                        f"[ChatAgent] Interrupted before tool '{tc['name']}', "
                        f"skipping remaining {len(tool_calls)} tool call(s)"
                    )
                    interrupted_during_tools = True
                    break

                args = {}
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    logger.warning(
                        f"[ChatAgent] Failed to parse tool args: {tc['arguments'][:100]}"
                    )

                result = registry.execute(tc["name"], args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result.to_content_str(),
                })
                logger.info(
                    f"[ChatAgent]   tool={tc['name']} → "
                    f"{result.to_content_str()[:120]}"
                )

            if interrupted_during_tools:
                logger.info("[ChatAgent] Agent loop aborted due to interrupt during tool execution")
                return None

            if context.pending_confirmations:
                context.pending_confirmations.tick_round()
                nag = context.pending_confirmations.get_nag_reminder()
                if nag:
                    messages.append({"role": "user", "content": nag})
                    logger.info("[ChatAgent] Injected pending-confirmations nag reminder")

        logger.warning(
            f"[ChatAgent] Agent loop reached max rounds ({max_rounds}), "
            "forcing text response"
        )
        return full_text

    def _stream_response(
        self,
        context: ChatAgentContext,
        response,
        output_definition,
        streamer,
        stream_key: Optional[str],
    ) -> tuple:
        """Stream an LLM response, accumulating text and tool_calls.

        Returns (full_text, tool_calls_list, cancelled).
        tool_calls_list is a list of dicts: [{"id", "name", "arguments"}, ...]
        """
        full_text = ""
        tool_calls_accum: Dict[int, dict] = {}
        cancelled = False

        for chunk in response:
            if stream_key and stream_key not in context.active_stream_keys:
                cancelled = True
                try:
                    response.close()
                except Exception:
                    pass
                break

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                full_text += delta.content
                output = DataBundle(output_definition)
                output.set_main_data(delta.content)
                streamer.stream_data(output)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = tool_calls_accum[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

        tool_calls_list = [
            tool_calls_accum[idx]
            for idx in sorted(tool_calls_accum.keys())
        ] if tool_calls_accum else []

        return full_text, tool_calls_list, cancelled

    def _check_compact(self, context: ChatAgentContext):
        """Check and trigger automatic dialogue context compaction."""
        if context.memory and context.memory.should_compact() and context.llm_client:
            compact_model = (
                context.config.context_compact.compact_model
                or context.config.llm_model
            )
            env_state = ""
            if context.memory:
                env_state = context.memory.perception_buffer.get_state_summary()
            task_brief = ""
            if context.task_mirror:
                task_brief = context.task_mirror.get_active_brief()
            context.memory.check_and_compact(
                context.llm_client, compact_model,
                task_brief=task_brief, env_state=env_state,
            )

    # ── Event Helper Methods ──

    def _should_skip_event(
        self, context: ChatAgentContext, event: EnvironmentEvent, cooldown: float
    ) -> bool:
        last = context.responded_events.get(event.event_type)
        if last is None:
            return False
        return (time.time() - last) <= cooldown

    def _process_pending_events(self, context: ChatAgentContext):
        """Merge all valid pending events into a single LLM invocation."""
        if not context.pending_events or not context.output_definitions:
            return

        max_event_age = 15.0
        now = time.time()

        valid_events: List[EnvironmentEvent] = []
        valid_cfgs: List[Optional[EventTriggerConfig]] = []

        while context.pending_events:
            event = context.pending_events.pop(0)
            age = now - event.timestamp if event.timestamp > 0 else 0.0

            if age > max_event_age:
                logger.info(f"[ChatAgent] Discarding expired event: {event.event_type} (age={age:.1f}s)")
                continue

            trigger_cfg = context.config.proactive.event_triggers.get(event.event_type)
            if trigger_cfg and not trigger_cfg.enabled:
                continue

            cooldown = trigger_cfg.cooldown if trigger_cfg else 30.0
            if self._should_skip_event(context, event, cooldown):
                continue

            valid_events.append(event)
            valid_cfgs.append(trigger_cfg)

        if valid_events:
            event_names = ", ".join(ev.event_type for ev in valid_events)
            logger.info(
                f"[ChatAgent] Consolidated processing of {len(valid_events)} pending events: {event_names}"
            )
            self._handle_proactive_response(
                context, valid_events, context.output_definitions, valid_cfgs
            )

    @staticmethod
    def _event_importance(event: EnvironmentEvent) -> float:
        if event.urgency == "critical":
            return 0.9
        elif event.urgency == "high":
            return 0.7
        return 0.5

    # ── Idle Trigger ──

    def _idle_trigger_loop(self, context: ChatAgentContext):
        """Background thread: detect user idle and trigger proactive dialogue."""
        check_interval = 2.0
        approval_grace_seconds = 1

        while not context._idle_stop.is_set():
            woken = context._proactive_wake.wait(timeout=check_interval)
            if woken:
                context._proactive_wake.clear()

            if context._idle_stop.is_set():
                break
            if context.is_generating or not context.output_definitions:
                continue

            elapsed = time.time() - context.last_interaction_time
            pc_cfg = context.config.proactive.pending_confirmation_trigger

            if (
                woken
                and pc_cfg.enabled
                and context.pending_confirmations
                and context.pending_confirmations.has_pending()
                and not self._should_skip_event(
                    context,
                    EnvironmentEvent(event_type="pending_confirmation"),
                    pc_cfg.cooldown,
                )
            ):
                time.sleep(approval_grace_seconds)
                if context.is_generating:
                    continue
                logger.info(
                    "[ChatAgent] Approval immediate proactive trigger (proactive_wake)"
                )
                self._fire_pending_confirmation(context, pc_cfg)
                continue

            if (
                pc_cfg.enabled
                and context.pending_confirmations
                and context.pending_confirmations.has_pending()
                and elapsed >= pc_cfg.idle_seconds
                and not self._should_skip_event(
                    context,
                    EnvironmentEvent(event_type="pending_confirmation"),
                    pc_cfg.cooldown,
                )
            ):
                logger.info(
                    f"[ChatAgent] Pending confirmation proactive trigger: "
                    f"elapsed={elapsed:.0f}s >= {pc_cfg.idle_seconds:.0f}s"
                )
                self._fire_pending_confirmation(context, pc_cfg)
                continue

            if context._idle_triggered:
                continue

            idle_cfg = context.config.proactive.idle_trigger
            if not idle_cfg.enabled:
                continue

            current_mode = (
                context.memory.session_mode if context.memory else "chitchat"
            )
            threshold = idle_cfg.mode_overrides.get(
                current_mode, idle_cfg.idle_seconds
            )

            if elapsed >= threshold:
                logger.info(
                    f"[ChatAgent] Idle trigger: "
                    f"{elapsed:.0f}s >= {threshold:.0f}s (mode={current_mode})"
                )
                idle_event = EnvironmentEvent(
                    event_type="idle",
                    description="User has been quiet for a while",
                    confidence=1.0,
                    urgency="low",
                    timestamp=time.time(),
                )
                idle_trigger_cfg = EventTriggerConfig(
                    hint=idle_cfg.hint,
                    cooldown=threshold,
                )
                self._handle_proactive_response(
                    context,
                    [idle_event],
                    context.output_definitions,
                    [idle_trigger_cfg],
                )
                context._idle_triggered = True

    def _fire_pending_confirmation(
        self,
        context: ChatAgentContext,
        pc_cfg: PendingConfirmationTriggerConfig,
    ):
        """Trigger a pending-confirmation proactive response."""
        pc_event = EnvironmentEvent(
            event_type="pending_confirmation",
            description=context.pending_confirmations.render(),
            confidence=1.0,
            urgency="high",
            timestamp=time.time(),
        )
        pc_trigger_cfg = EventTriggerConfig(
            hint=pc_cfg.hint,
            cooldown=pc_cfg.cooldown,
        )
        self._handle_proactive_response(
            context, [pc_event], context.output_definitions, [pc_trigger_cfg]
        )

    # ── OC Bridge Initialization ──

    @staticmethod
    def _init_oc_bridge(context: ChatAgentContext):
        """Delegate to ``oc_bridge.oc_bridge_init`` for all OC component wiring."""
        from handlers.agent.oc_bridge.oc_bridge_init import init_oc_bridge
        init_oc_bridge(context)

    # ── ToolRegistry Build ──

    @staticmethod
    def _build_tool_registry(config: ChatAgentConfig) -> ToolRegistry:
        registry = ToolRegistry()

        if config.tool_use.enabled and config.tool_use.register_demo_tools:
            from handlers.agent.tools.demo_tools import GetCurrentTimeTool, GetSystemInfoTool
            registry.register(GetCurrentTimeTool())
            registry.register(GetSystemInfoTool())

        logger.info(
            f"[ChatAgent] ToolRegistry initialized with {len(registry.tool_names)} tools: "
            f"{registry.tool_names}"
        )
        return registry

    # ── Debug Logging ──

    def _log_incomplete_work_summary(self, context: "ChatAgentContext"):
        """Print a one-shot INFO summary of unfinished work (queue, approvals, mirror)."""
        lines: List[str] = []

        if context.task_queue and context.task_queue.size > 0:
            for n in context.task_queue.peek():
                summary = (n.result_summary or "").replace("\n", " ")[:120]
                lines.append(
                    f"  [task_queue] {n.task_id} | {n.status} | {summary}"
                )

        if context.pending_confirmations and context.pending_confirmations.has_pending():
            for item in context.pending_confirmations.get_pending_items():
                txt = (item.text or "").replace("\n", " ")[:120]
                lines.append(
                    f"  [pending_confirm] id={item.id} | {txt}"
                )

        if context.task_mirror:
            active = context.task_mirror.get_active_tasks()
            for t in active:
                brief = (t.brief or "").replace("\n", " ")[:80]
                lines.append(
                    f"  [task_mirror] {t.task_id} | {t.status} | {t.title} | {brief}"
                )

        if lines:
            logger.debug(
                "[ChatAgent] Unfinished work items ({} items):\n{}",
                len(lines),
                "\n".join(lines),
            )
        else:
            logger.debug("[ChatAgent] Unfinished work items: None")

    def _debug_log_prompt(
        self,
        context: "ChatAgentContext",
        compiled,
    ):
        """Log environment state, full system prompt (line-by-line), messages, and tool list."""
        sys_text = compiled.system_message
        env_match = re.search(
            r"^<environment-state>\s*\n",
            sys_text,
            re.MULTILINE,
        )
        if env_match:
            env_start = env_match.start()
            env_end = sys_text.find("</environment-state>", env_start)
            env_section = (
                sys_text[env_start: env_end + len("</environment-state>")]
                if env_end >= 0
                else sys_text[env_start:]
            )
            logger.debug("[ChatAgent] ─── Environment State (L3, real block) ───")
            for line in env_section.splitlines():
                logger.debug("[ChatAgent]   {}", line)
        else:
            logger.debug(
                "[ChatAgent] ─── Environment State (L3): (empty — no perception snapshot or not injected) ───"
            )

        logger.debug(
            "[ChatAgent] ─── System Message (full, {} chars, L1–L3) ───",
            len(sys_text),
        )
        for lineno, line in enumerate(sys_text.splitlines(), start=1):
            logger.debug("[ChatAgent]   sys {:4d} | {}", lineno, line)

        use_tools = (
            context.tool_registry is not None
            and context.tool_registry.has_tools()
            and context.config.tool_use.enabled
        )
        if use_tools:
            schemas = context.tool_registry.get_schemas()
            logger.debug(
                "[ChatAgent] ─── Tool definitions (OpenAI `tools` param, NOT in system prompt) ─── count={}",
                len(schemas),
            )
            for s in schemas:
                fn = s.get("function", {})
                name = fn.get("name", "?")
                desc = (fn.get("description") or "").replace("\n", " ")
                logger.debug("[ChatAgent]   tool: {} | {}", name, desc[:300])
        else:
            logger.debug(
                "[ChatAgent] ─── Tools: disabled or none (not passed to LLM) ───"
            )

        logger.debug("[ChatAgent] ─── Chat messages (full_messages, incl. L4) ───")
        for i, msg in enumerate(compiled.full_messages):
            role = msg.get("role", "?")
            content = msg.get("content")
            if content is None or content == "":
                logger.debug("[ChatAgent]   [{}] {}: (no content)", i, role)
                continue
            body = content if isinstance(content, str) else str(content)
            logger.debug("[ChatAgent]   [{}] {} ({} chars) ───", i, role, len(body))
            for line in body.splitlines():
                logger.debug("[ChatAgent]       {}", line)
        logger.debug("[ChatAgent] ─── End prompt debug ───")

    # ── PromptCompiler Construction ──

    @staticmethod
    def _build_compiler(config: ChatAgentConfig) -> PromptCompiler:
        layer_configs = {
            LAYER_PERSONA_SNAPSHOT: PromptLayerConfig(
                section_header="[Persona]",
                max_chars=config.persona_max_chars,
            ),
            LAYER_ENVIRONMENT_STATE: PromptLayerConfig(
                max_chars=config.perception_max_chars,
            ),
        }
        return PromptCompiler(
            stable_core=(
                f"{config.stable_core.rstrip()}\n\n"
                f"{MANDATORY_DELEGATION_POLICY}\n\n"
                f"{REALTIME_AND_TRUTH_POLICY}"
            ),
            persona_snapshot=config.persona_snapshot,
            layer_configs=layer_configs,
            max_dialogue_turns=config.compiler_dialogue_turns,
        )

    # ── Lifecycle ──

    def destroy_context(self, context: HandlerContext):
        context = cast(ChatAgentContext, context)
        context._idle_stop.set()
        if context.oc_channel_client:
            try:
                context.oc_channel_client.stop()
            except Exception:
                pass
        if context.oc_mcp_client:
            try:
                context.oc_mcp_client.stop()
            except Exception:
                pass
        if context.memory:
            context.memory.destroy()
        context.compiler = None
        context.tool_registry = None
        context.oc_mcp_client = None
        context.oc_channel_client = None
        context.persona_mgr = None
        context.task_queue = None
        context.task_mirror = None
        context.pending_confirmations = None
        logger.info(f"ChatAgentContext destroyed for session {context.session_id}")
