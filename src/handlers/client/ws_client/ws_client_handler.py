"""
WebSocket Client Handler
WebSocket-based session endpoint digital human interaction processor.
"""
from typing import Any, Dict, Optional, cast

import gradio
from fastapi import FastAPI, WebSocketDisconnect
from loguru import logger
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket, WebSocketState

from chat_engine.common.client_handler_base import ClientHandlerBase, ClientSessionDelegate
from chat_engine.common.handler_base import HandlerDataInfo, HandlerDetail, HandlerBaseInfo
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_stream_config import ChatStreamConfig
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalType
from chat_engine.data_models.engine_channel_type import EngineChannelType
from chat_engine.data_models.runtime_data.data_bundle import DataBundleDefinition, DataBundleEntry, VariableSize

from .ws_input_delegate import WsInputSessionDelegate
from service.frontend_service import register_frontend

# ============================================================================
# Configuration Model
# ============================================================================

class WsClientConfigModel(HandlerBaseConfigModel, BaseModel):
    """WebSocket Client Configuration."""
    connection_ttl: int = Field(default=900, description="Maximum connection duration (seconds)")
    heartbeat_timeout: int = Field(default=30, description="Heartbeat timeout duration (seconds)")

# ============================================================================
# Handler Context
# ============================================================================

class WsClientContext(HandlerContext):
    """WebSocket Client Context."""
    
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: Optional[WsClientConfigModel] = None
        self.client_session_delegate: Optional[WsInputSessionDelegate] = None

# ============================================================================
# WebSocket Client Handler
# ============================================================================

class WsClientHandler(ClientHandlerBase):
    """
    WebSocket Client Handler
    
    Provides a single WebSocket session endpoint:
    - /ws/session/{session_id}: Client responsible for input upload & Motion Data downstream consumption
    """
    
    def __init__(self):
        super().__init__()
        self.engine_config: Optional[ChatEngineConfigModel] = None
        self.handler_config: Optional[WsClientConfigModel] = None
        
        # Data definitions
        self.input_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
        self.output_bundle_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}
    
    def get_handler_info(self) -> HandlerBaseInfo:
        """Get Handler Info."""
        return HandlerBaseInfo(
            config_model=WsClientConfigModel,
            client_session_delegate_class=WsInputSessionDelegate,
        )
    
    def prepare_data_definitions(self):
        """Prepare data definitions."""
        # Input definitions (Client upload)
        # Audio definition
        audio_input_definition = DataBundleDefinition()
        audio_input_definition.add_entry(DataBundleEntry.create_audio_entry(
            "mic_audio",
            1,  # mono
            16000,  # 16kHz
        ))
        audio_input_definition.lockdown()
        self.input_bundle_definitions[EngineChannelType.AUDIO] = audio_input_definition
        
        # Video definition
        video_input_definition = DataBundleDefinition()
        video_input_definition.add_entry(DataBundleEntry.create_framed_entry(
            "camera_video",
            [VariableSize(), VariableSize(), VariableSize(), 3],  # [H, W, C, 3]
            0,  # time_axis
            30  # fps
        ))
        video_input_definition.lockdown()
        self.input_bundle_definitions[EngineChannelType.VIDEO] = video_input_definition
        
        # Text definition
        text_input_definition = DataBundleDefinition()
        text_input_definition.add_entry(DataBundleEntry.create_text_entry(
            "human_text",
        ))
        text_input_definition.lockdown()
        self.input_bundle_definitions[EngineChannelType.TEXT] = text_input_definition
        
        # Output definitions (Engine output to Client)
        # Identical to input definitions as client may require echo
        self.output_bundle_definitions = self.input_bundle_definitions.copy()
        
        logger.info("Data definitions prepared")
    
    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[HandlerBaseConfigModel] = None):
        """Load configuration."""
        self.engine_config = engine_config
        
        if handler_config is None or not isinstance(handler_config, WsClientConfigModel):
            handler_config = WsClientConfigModel()
        
        self.handler_config = handler_config
        
        # Prepare data definitions
        self.prepare_data_definitions()
        
        logger.info(f"WsClientHandler loaded with config: {self.handler_config}")
    def on_setup_app(self, app: FastAPI, ui: gradio.blocks.Block, parent_block: Optional[gradio.blocks.Block] = None):
        """Set up FastAPI routes."""
        
        @app.websocket("/ws/session/{session_id}")
        async def ws_session_endpoint(websocket: WebSocket, session_id: str):
            """Single endpoint session - responsible for uploading input & receiving Motion Data."""
            await websocket.accept()
            logger.info(f"Session WebSocket connected: session_id={session_id}")
            
            should_stop = False
            try:
                # Find or create session
                session_delegate = self.handler_delegate.find_session_delegate(session_id)
                
                if session_delegate is None:
                    # Create new session
                    logger.info(f"Creating new session: {session_id}")
                    session_delegate = self.handler_delegate.start_session(session_id)
                
                if not isinstance(session_delegate, WsInputSessionDelegate):
                    logger.error(f"Invalid session delegate type: {type(session_delegate)}")
                    await websocket.close(code=1003, reason="Invalid session")
                    return
                
                # Serve WebSocket
                should_stop = await session_delegate.serve_websocket(websocket)
            
            except WebSocketDisconnect:
                logger.info(f"Session WebSocket disconnected: session_id={session_id}")
            except Exception as e:
                logger.error(f"Error in session WebSocket: {e}")
            finally:
                # Clean up session if main connection disconnects or session ends
                if should_stop:
                    try:
                        self.handler_delegate.stop_session(session_id)
                        logger.info(f"Session stopped: {session_id}")
                    except Exception as e:
                        logger.error(f"Error stopping session: {e}")
                
                # Ensure connection is closed
                try:
                    if not websocket.client_state == WebSocketState.DISCONNECTED:
                        await websocket.close()
                except Exception:
                    pass
        
        self.register_additional_routes(app)

        def init_config_provider():
            return self.build_frontend_init_config()

        register_frontend(
            app=app,
            ui=ui,
            parent_block=parent_block,
            init_config=init_config_provider,
        )
        logger.info("WebSocket route registered: /ws/session/{session_id}")

    def register_additional_routes(self, app: FastAPI):
        """Hook for subclasses to register additional FastAPI routes."""
        return

    def get_additional_init_config(self) -> Dict[str, Any]:
        """Hook for subclasses to extend init config payload."""
        return {}

    def build_frontend_init_config(self) -> Dict[str, Any]:
        base_config: Dict[str, Any] = {
            "chat_mode": "ws",
            "ws_session_route": "/ws/session",
            "track_constraints": {
                "audio": {
                    "sampleRate": 16000,
                    "channelCount": 1,
                    "autoGainControl": False,
                    "noiseSuppression": False,
                    "echoCancellation": True,
                }
            }
        }
        additional = self.get_additional_init_config()
        if additional:
            base_config.update(additional)
        return base_config
    
    
    def create_context(self, session_context: SessionContext,
                       handler_config: Optional[HandlerBaseConfigModel] = None) -> HandlerContext:
        """Create Handler Context."""
        if not isinstance(handler_config, WsClientConfigModel):
            handler_config = WsClientConfigModel()
        
        context = WsClientContext(session_context.session_info.session_id)
        context.config = handler_config
        return context
    
    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        """Start Context."""
        pass
    
    def on_setup_session_delegate(self, session_context: SessionContext, handler_context: HandlerContext,
                                  session_delegate: ClientSessionDelegate):
        """Set up session delegate."""
        handler_context = cast(WsClientContext, handler_context)
        session_delegate = cast(WsInputSessionDelegate, session_delegate)
        
        # Set attributes on session delegate
        session_delegate.session_id = session_context.session_info.session_id
        session_delegate.clock = session_context.get_clock()
        session_delegate.data_submitter = handler_context.data_submitter
        session_delegate.signal_emitter = handler_context.signal_emitter
        session_delegate.input_data_definitions = self.input_bundle_definitions
        session_delegate.shared_states = session_context.shared_states
        session_delegate.heartbeat_timeout = self.handler_config.heartbeat_timeout
        session_delegate.session_history = session_context.session_history
        session_delegate.stream_manager = handler_context.stream_manager
        
        # Save reference
        handler_context.client_session_delegate = session_delegate
        
        logger.info(f"Session delegate setup completed for session {session_context.session_info.session_id}")
    
    def create_handler_detail(self, _session_context, _handler_context):
        """Create Handler Detail."""
        # Inputs: Data output from engine to client
        inputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                input_priority=-1  # Higher priority to receive data before ONCE mode handlers
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                input_priority=-1  # Ensure client receives all text data
            ),
            ChatDataType.AVATAR_MOTION_DATA: HandlerDataInfo(
                type=ChatDataType.AVATAR_MOTION_DATA,
                input_priority=-1  # Ensure client receives all motion data
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                input_priority=-1  # Ensure client receives ASR recognized text
            ),
        }
        
        # Outputs: Data uploaded by client to engine
        _no_link = ChatStreamConfig(cancelable=False, auto_link_input=False)
        outputs = {
            ChatDataType.MIC_AUDIO: HandlerDataInfo(
                type=ChatDataType.MIC_AUDIO,
                definition=self.output_bundle_definitions[EngineChannelType.AUDIO],
                output_stream_config=_no_link,
            ),
            ChatDataType.CAMERA_VIDEO: HandlerDataInfo(
                type=ChatDataType.CAMERA_VIDEO,
                definition=self.output_bundle_definitions[EngineChannelType.VIDEO],
                output_stream_config=_no_link,
            ),
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=self.output_bundle_definitions[EngineChannelType.TEXT],
                output_stream_config=_no_link,
            ),
        }
        

        return HandlerDetail(
            inputs=inputs,
            outputs=outputs,
        )
    
    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        """Get Handler Detail."""
        return self.create_handler_detail(session_context, context)
    
    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        """
        Handle engine output data.
        Route data to corresponding output queue for WebSocket transmission.
        """
        context = cast(WsClientContext, context)
        
        if context.client_session_delegate is None:
            return
        
        # Route to corresponding queue based on channel type
        channel_type = inputs.type.channel_type
        data_queue = context.client_session_delegate.output_queues.get(channel_type)
        
        if data_queue is not None:
            data_queue.put_nowait(inputs)
            logger.debug(f"Routed {inputs.type} to {channel_type} queue")
    
    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        """Handle signals."""
        logger.info(f"Received signal: {signal.type} from {signal.source_type} on stream {signal.related_stream}")
        context = cast(WsClientContext, context)
        if context.client_session_delegate is None:
            return
        
        # Handle interrupt signal: reset Opus encoder to flush residual buffer
        if signal.type == ChatSignalType.INTERRUPT:
            if context.client_session_delegate._opus_encoder is not None:
                context.client_session_delegate._opus_encoder.reset()
                logger.debug("Opus encoder reset due to interrupt signal from engine")
        
        context.client_session_delegate.signal_to_client_queue.put_nowait(signal)

    def destroy_context(self, context: HandlerContext):
        """Destroy Context."""
        context = cast(WsClientContext, context)
        
        if context.client_session_delegate is not None:
            context.client_session_delegate.quit.set()
            context.client_session_delegate.clear_data()
        
        logger.info(f"Context destroyed for session {context.session_id}")

