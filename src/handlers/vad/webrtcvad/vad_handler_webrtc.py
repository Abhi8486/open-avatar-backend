import enum
import collections
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

from chat_engine.common.handler_base import HandlerBase, HandlerDetail, HandlerDataInfo, HandlerBaseInfo
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_engine_config_data import HandlerBaseConfigModel, ChatEngineConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class WebRTCVADConfigModel(HandlerBaseConfigModel, BaseModel):
    aggressiveness: int = Field(default=3, description="0 is least aggressive about filtering out non-speech, 3 is most aggressive.")
    frame_duration_ms: int = Field(default=30, description="Frame duration in ms (10, 20, or 30)")
    padding_duration_ms: int = Field(default=300, description="Amount of history to keep to determine if speech started/stopped")


class WebRTCVADContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = WebRTCVADConfigModel()
        self.vad = None
        self.sample_rate = 16000
        self.frame_length = int(self.sample_rate * (30 / 1000.0)) # default 30ms
        self.audio_buffer = np.array([], dtype=np.int16)
        
        self.ring_buffer = collections.deque(maxlen=10) # ~300ms of frames
        self.triggered = False
        self.voiced_frames = []
        self.input_enabled = True


class HandlerAudioVAD(HandlerBase):
    def __init__(self, config: dict):
        super().__init__(config)
        if webrtcvad is None:
            logger.error("webrtcvad package is not installed! Please run: pip install webrtcvad")

    def _get_context_class(self):
        return WebRTCVADContext

    def create_context(self, session_context: SessionContext) -> HandlerContext:
        context = super().create_context(session_context)
        context.config = WebRTCVADConfigModel(**self.config_dict)
        if webrtcvad is not None:
            context.vad = webrtcvad.Vad(context.config.aggressiveness)
        context.frame_length = int(context.sample_rate * (context.config.frame_duration_ms / 1000.0))
        num_padding_frames = int(context.config.padding_duration_ms / context.config.frame_duration_ms)
        context.ring_buffer = collections.deque(maxlen=num_padding_frames)
        return context

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(config_model=WebRTCVADConfigModel)

    def load(self, engine_config: ChatEngineConfigModel, handler_config = None):
        pass

    def start_context(self, session_context: SessionContext, handler_context: HandlerContext):
        pass

    def destroy_context(self, context: HandlerContext):
        pass

    def get_handler_detail(self, session_context: SessionContext, context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 16000))
        return HandlerDetail(
            inputs=[HandlerDataInfo(type=ChatDataType.MIC_AUDIO)],
            outputs=[HandlerDataInfo(type=ChatDataType.HUMAN_AUDIO, definition=definition)],
            signal_filters=[
                SignalFilterRule(ChatSignalType.STREAM_BEGIN, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_END, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.CLIENT_PLAYBACK),
            ]
        )

    def handle(self, context: HandlerContext, inputs: ChatData, output_definitions: dict):
        if not context.input_enabled or webrtcvad is None:
            return

        audio = inputs.data.get_main_data()
        if audio is None:
            return
            
        # Ensure int16
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
            
        audio = audio.flatten()
        context.audio_buffer = np.concatenate((context.audio_buffer, audio))
        
        while len(context.audio_buffer) >= context.frame_length:
            frame = context.audio_buffer[:context.frame_length]
            context.audio_buffer = context.audio_buffer[context.frame_length:]
            
            is_speech = context.vad.is_speech(frame.tobytes(), context.sample_rate)
            
            if not context.triggered:
                context.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in context.ring_buffer if speech])
                
                # If >90% of the ring buffer is speech, trigger!
                if num_voiced > 0.9 * context.ring_buffer.maxlen:
                    context.triggered = True
                    logger.info("WebRTCVAD: Start of speech detected!")
                    # Yield start signal
                    self.emit_signal(context, ChatSignal(ChatSignalType.STREAM_BEGIN, ChatSignalSourceType.HANDLER, ChatDataType.HUMAN_AUDIO))
                    
                    # Yield all buffered frames
                    for f, s in context.ring_buffer:
                        context.voiced_frames.append(f)
                    context.ring_buffer.clear()
            else:
                context.voiced_frames.append(frame)
                context.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in context.ring_buffer if not speech])
                
                # If >90% of ring buffer is silence, untrigger!
                if num_unvoiced > 0.9 * context.ring_buffer.maxlen:
                    context.triggered = False
                    logger.info("WebRTCVAD: End of speech detected!")
                    
                    # Send collected audio
                    complete_audio = np.concatenate(context.voiced_frames)
                    
                    # Reshape for output
                    complete_audio = np.expand_dims(complete_audio, axis=0)
                    
                    bundle = DataBundle()
                    bundle.entries["human_audio"] = DataBundleEntry(complete_audio, 1, context.sample_rate)
                    
                    data = ChatData(ChatDataType.HUMAN_AUDIO, bundle)
                    self.emit(context, data)
                    
                    self.emit_signal(context, ChatSignal(ChatSignalType.STREAM_END, ChatSignalSourceType.HANDLER, ChatDataType.HUMAN_AUDIO))
                    
                    context.ring_buffer.clear()
                    context.voiced_frames = []

    def handle_signal(self, context: HandlerContext, signal: ChatSignal):
        if signal.signal_type == ChatSignalType.STREAM_BEGIN:
            context.input_enabled = False
            logger.info("WebRTCVAD: Paused listening (Simplex mode)")
        elif signal.signal_type in (ChatSignalType.STREAM_END, ChatSignalType.STREAM_CANCEL):
            context.input_enabled = True
            logger.info("WebRTCVAD: Resumed listening (Simplex mode)")
