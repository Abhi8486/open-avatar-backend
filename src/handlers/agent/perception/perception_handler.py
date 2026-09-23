"""
Perception Agent Handler

Processes video streams, generates layered visual context, and detects environmental events.
"""
import json
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional, cast

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry

from handlers.agent.agent_data_models import PerceptionData, EnvironmentEvent
from handlers.agent.perception.vision_model_interface import (
    VisionModelInterface,
    MockVisionModel,
    OpenAIVisionModel,
    AsyncPerceptionManager,
)


class PerceptionConfig(HandlerBaseConfigModel, BaseModel):
    """Perception Handler Configuration"""
    summary_interval: float = Field(default=3.0, description="Interval for generating visual summary (seconds)")
    
    max_buffer_frames: int = Field(default=30, description="Maximum buffer frames")
    
    key_frame_strategy: str = Field(default="interval", description="Keyframe selection strategy: interval, motion, all")
    key_frame_interval: int = Field(default=10, description="Keyframe interval (for interval strategy)")
    
    vision_model_type: str = Field(default="mock", description="Vision model type: mock, qwen_vl, openai")

    llm_model: str = Field(default="qwen-plus-vl", description="Vision model name")
    api_key: Optional[str] = Field(default=None, description="API Key (defaults to env var)")
    api_url: Optional[str] = Field(default=None, description="API URL")
    max_frames: int = Field(default=4, description="Maximum frames to send per request")
    
    enable_event_detection: bool = Field(default=True, description="Whether to enable event detection")
    event_detection_interval: int = Field(default=5, description="Event detection interval (every N frames)")
    
    max_concurrent_requests: int = Field(default=3, description="Maximum concurrent LLM requests")

    vlm_system_prompt: Optional[str] = Field(default=None, description="VLM system prompt override")


@dataclass
class PerceptionContext(HandlerContext):
    """Perception Handler Context"""
    
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[PerceptionConfig] = None
        
        self.frame_buffer: List[np.ndarray] = []
        self.frame_count: int = 0
        
        self.last_summary_time: float = 0.0
        
        self.previous_frame: Optional[np.ndarray] = None
        
        self.current_perception: Optional[PerceptionData] = None
        
        self.vision_model: Optional[VisionModelInterface] = None
        
        self.async_manager: Optional[AsyncPerceptionManager] = None
        
        self.fps_start_time: float = 0.0
        self.fps_frame_count: int = 0
        self.current_fps: float = 0.0
        self.fps_update_interval: float = 2.0
        
        self.perception_round: int = 0
        
        self.output_definitions: Optional[Dict[ChatDataType, HandlerDataInfo]] = None
        
        self.last_event_times: Dict[str, float] = {}
        self.event_dedup_interval: float = 5.0
        
        self.last_frame_time: float = 0.0
        self.frame_stall_warned: bool = False
        self.frame_stall_threshold: float = 10.0
        self.frame_resumed_after_stall: bool = False
        self._heartbeat_stop: threading.Event = threading.Event()


class PerceptionHandler(HandlerBase, ABC):
    """
    Perception Agent Handler
    
    Responsibilities:
    1. Receive CAMERA_VIDEO stream
    2. Periodically generate layered visual context (PERCEPTION_CONTEXT)
    3. Real-time detection of environmental events (emitted via Signal as ENVIRONMENT_EVENT)
    """
    
    def __init__(self):
        super().__init__()
        self.output_definition: Optional[DataBundleDefinition] = None
    
    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=PerceptionConfig,
        )
    
    def load(self, engine_config: ChatEngineConfigModel, 
             handler_config: Optional[HandlerBaseConfigModel] = None):
        """Load Handler."""
        self.output_definition = DataBundleDefinition()
        self.output_definition.add_entry(
            DataBundleEntry(
                name="perception_data",
            )
        )
        logger.info("PerceptionHandler loaded")
    
    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        """Create session context."""
        context = PerceptionContext(session_context.session_info.session_id)
        
        if isinstance(handler_config, PerceptionConfig):
            context.config = handler_config
        else:
            context.config = PerceptionConfig()
        
        context.vision_model = self._create_vision_model(context.config)
        
        context.async_manager = AsyncPerceptionManager(
            vision_model=context.vision_model,
            max_workers=context.config.max_concurrent_requests,
            on_result_callback=lambda round_id, data: self._on_async_result(context, round_id, data),
        )
        
        logger.info(f"PerceptionContext created for session {context.session_id} "
                   f"(max_concurrent_requests={context.config.max_concurrent_requests})")
        return context
    
    def _create_vision_model(self, config: PerceptionConfig) -> VisionModelInterface:
        """Create vision model based on configuration."""
        model_type = config.vision_model_type
        if model_type == "mock":
            return MockVisionModel()
        if model_type == "openai":
            return OpenAIVisionModel(
                model_name=config.llm_model,
                api_key=config.api_key,
                api_url=config.api_url,
                max_frames=config.max_frames,
                system_prompt=config.vlm_system_prompt,
            )
        else:
            logger.warning(f"Unknown vision model type: {model_type}, using mock")
            return MockVisionModel()
    
    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        """Start context."""
        context = cast(PerceptionContext, handler_context)
        if context.vision_model:
            context.vision_model.warmup()
        
        heartbeat_thread = threading.Thread(
            target=self._frame_heartbeat_monitor,
            args=(context,),
            daemon=True,
            name=f"perception-heartbeat-{context.session_id}",
        )
        heartbeat_thread.start()
    
    def _frame_heartbeat_monitor(self, context: PerceptionContext):
        """Background thread: Periodically check if camera frames are arriving."""
        check_interval = 5.0
        while not context._heartbeat_stop.wait(timeout=check_interval):
            if context.last_frame_time == 0.0:
                continue
            
            gap = time.time() - context.last_frame_time
            if gap >= context.frame_stall_threshold and not context.frame_stall_warned:
                context.frame_stall_warned = True
                logger.warning(
                    f"[Perception] ⚠️ Camera frame stream stalled! No new frames received for {gap:.1f}s "
                    f"(Threshold: {context.frame_stall_threshold}s, "
                    f"Processed frames: {context.frame_count})"
                )
    
    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        """Define inputs and outputs."""
        perception_definition = DataBundleDefinition()
        perception_definition.add_entry(
            DataBundleEntry(
                name="perception_data",
            )
        )
        
        event_definition = DataBundleDefinition()
        event_definition.add_entry(
            DataBundleEntry(
                name="event_data",
            )
        )
        
        return HandlerDetail(
            inputs={
                ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                    type=ChatDataType.CAMERA_VIDEO,
                ),
            },
            outputs={
                ChatDataType.PERCEPTION_CONTEXT: HandlerDataInfo(
                    type=ChatDataType.PERCEPTION_CONTEXT,
                    definition=perception_definition,
                ),
                ChatDataType.ENVIRONMENT_EVENT: HandlerDataInfo(
                    type=ChatDataType.ENVIRONMENT_EVENT,
                    definition=event_definition,
                ),
            },
        )
    
    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """Handle video frames."""
        context = cast(PerceptionContext, context)
        
        if inputs.type != ChatDataType.CAMERA_VIDEO:
            logger.debug(f"[Perception] Skip non-video input: {inputs.type}")
            return
        
        frame = inputs.data.get_main_data()
        if frame is None:
            logger.warning("[Perception] Received empty frame data")
            return
        
        current_time = time.time()
        
        if context.frame_stall_warned:
            gap = current_time - context.last_frame_time
            logger.warning(
                f"[Perception] 📡 Camera frame stream resumed! Stall duration: {gap:.1f}s, "
                f"Total frames: {context.frame_count}"
            )
            context.frame_stall_warned = False
            context.frame_resumed_after_stall = True
        context.last_frame_time = current_time
        
        if context.fps_start_time == 0.0:
            context.fps_start_time = current_time
        context.fps_frame_count += 1
        
        fps_elapsed = current_time - context.fps_start_time
        if fps_elapsed >= context.fps_update_interval:
            context.current_fps = context.fps_frame_count / fps_elapsed
            logger.info(f"[Perception] 📊 Input FPS: {context.current_fps:.1f} FPS "
                       f"(Stat window: {fps_elapsed:.1f}s, Frames: {context.fps_frame_count})")
            context.fps_start_time = current_time
            context.fps_frame_count = 0
        
        if context.frame_count % 100 == 0:
            if isinstance(frame, np.ndarray):
                logger.debug(f"[Perception] Frame info: shape={frame.shape}, dtype={frame.dtype}, "
                           f"size={frame.size}, contiguous={frame.flags['C_CONTIGUOUS']}")
            else:
                logger.warning(f"[Perception] Frame is not ndarray, type={type(frame)}, value={str(frame)[:100]}")
        
        context.frame_count += 1
        
        if context.frame_count % 100 == 0:
            logger.info(f"[Perception] 📹 Processed {context.frame_count} frames, buffer: {len(context.frame_buffer)} frames, Current FPS: {context.current_fps:.1f}")
        
        self._add_to_buffer(context, frame)
        
        current_time = time.time()
        if current_time - context.last_summary_time >= context.config.summary_interval:
            logger.info(f"[Perception] ⏰ Trigger summary generation (Interval: {context.config.summary_interval}s)")
            self._generate_and_emit_perception(context, output_definitions)
            context.last_summary_time = current_time
        
        context.previous_frame = frame
    
    def _add_to_buffer(self, context: PerceptionContext, frame: np.ndarray):
        """Add frame to buffer."""
        should_add = False
        if context.config.key_frame_strategy == "all":
            should_add = True
        elif context.config.key_frame_strategy == "interval":
            if context.frame_count % context.config.key_frame_interval == 0:
                should_add = True
        
        if should_add:
            context.frame_buffer.append(frame)
        
        while len(context.frame_buffer) > context.config.max_buffer_frames:
            context.frame_buffer.pop(0)
    
    def _generate_and_emit_perception(self, context: PerceptionContext,
                                       output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """Generate and emit perception data (submit async task)."""
        if not context.frame_buffer:
            logger.debug("[Perception] Frame buffer empty, skipping summary generation")
            return
        
        if context.async_manager is None:
            logger.error("[Perception] AsyncPerceptionManager not initialized")
            return
        
        try:
            context.perception_round += 1
            round_id = context.perception_round
            round_tag = f"[Round-{round_id}]"
            
            buffer_size = len(context.frame_buffer)
            effective_fps = buffer_size / context.config.summary_interval if context.config.summary_interval > 0 else 0
            
            logger.info(f"[Perception] {round_tag} 🔍 Submitting async task:")
            logger.info(f"[Perception] {round_tag}   └─ Buffered keyframes: {buffer_size}")
            logger.info(f"[Perception] {round_tag}   └─ Input FPS: {context.current_fps:.1f} FPS")
            logger.info(f"[Perception] {round_tag}   └─ Keyframe effective FPS: {effective_fps:.1f} FPS (Saving 1 frame every {context.config.key_frame_interval} frames)")
            
            frames_snapshot = context.frame_buffer.copy()
            context.output_definitions = output_definitions
            
            submitted = context.async_manager.submit_task(
                round_id=round_id,
                frames=frames_snapshot,
            )
            
            if submitted:
                context.frame_buffer.clear()
            else:
                context.perception_round -= 1
                logger.warning(f"[Perception] {round_tag} ⚠️ Task submission failed, retaining buffer for next trigger")
            
        except Exception as e:
            logger.error(f"[Perception] ❌ Submitting perception task failed: {e}")
    
    def _on_async_result(self, context: PerceptionContext, round_id: int, 
                         perception: Optional[PerceptionData]):
        """Async task completion callback."""
        round_tag = f"[Round-{round_id}]"
        
        try:
            if perception is None:
                logger.warning(f"[Perception] {round_tag} ⚠️ Async task returned empty result, skipping emission")
                return
            
            context.current_perception = perception
            
            logger.info(f"[Perception] {round_tag} ✅ Async task completed:")
            logger.info(f"[Perception] {round_tag}   └─ Scene Summary: {perception.scene_summary}")
            logger.info(f"[Perception] {round_tag}   └─ User State: emotion={perception.user_state.emotion}, gaze={perception.user_state.gaze}")
            logger.info(f"[Perception] {round_tag}   └─ Scene Structure: location={perception.scene_structure.location}")
            
            output_definitions = context.output_definitions
            if output_definitions is None:
                logger.warning(f"[Perception] {round_tag} output_definitions not set, cannot emit")
                return
            
            output_def = output_definitions.get(ChatDataType.PERCEPTION_CONTEXT)
            if output_def and output_def.definition:
                output = DataBundle(output_def.definition)
                output.set_main_data(json.dumps(perception.to_dict(), ensure_ascii=False))
                context.submit_data((ChatDataType.PERCEPTION_CONTEXT, output))
                
                logger.info(f"[Perception] {round_tag} 📤 Emitted PERCEPTION_CONTEXT to Manager")
            
            self._emit_detected_events(context, perception, output_definitions, round_tag)
            
        except Exception as e:
            logger.error(f"[Perception] {round_tag} ❌ Processing async result failed: {e}")
    
    def _emit_detected_events(self, context: PerceptionContext, perception: PerceptionData,
                               output_definitions: Dict[ChatDataType, HandlerDataInfo], round_tag: str):
        """Check and emit detected interaction events."""
        triggerable_events = perception.get_triggerable_events()
        
        if not triggerable_events:
            return
        
        current_time = time.time()
        event_def = output_definitions.get(ChatDataType.ENVIRONMENT_EVENT)
        
        if not event_def or not event_def.definition:
            logger.warning(f"[Perception] {round_tag} ENVIRONMENT_EVENT output definition not set")
            return
        
        for detected_event in triggerable_events:
            last_time = context.last_event_times.get(detected_event.event_type, 0.0)
            if current_time - last_time < context.event_dedup_interval:
                logger.debug(f"[Perception] {round_tag} Skip duplicate event: {detected_event.event_type} "
                           f"({current_time - last_time:.1f}s since last)")
                continue
            
            context.last_event_times[detected_event.event_type] = current_time
            
            env_event = EnvironmentEvent.from_detected_event(detected_event, urgency="high")
            
            event_output = DataBundle(event_def.definition)
            event_output.set_main_data(json.dumps(env_event.to_dict(), ensure_ascii=False))
            context.submit_data((ChatDataType.ENVIRONMENT_EVENT, event_output))
            
            logger.info(f"[Perception] {round_tag} 📣 Detected interaction event: {detected_event.event_type} "
                       f"(confidence: {detected_event.confidence:.2f})")
            logger.info(f"[Perception] {round_tag} 📤 Emitted ENVIRONMENT_EVENT to Manager")
    
    def destroy_context(self, context: HandlerContext):
        """Destroy context."""
        context = cast(PerceptionContext, context)
        
        context._heartbeat_stop.set()
        
        if context.async_manager:
            context.async_manager.shutdown(wait=False, timeout=2.0)
            context.async_manager = None
        
        if context.vision_model:
            context.vision_model.cleanup()
        context.frame_buffer.clear()
        logger.info(f"PerceptionContext destroyed for session {context.session_id}")
