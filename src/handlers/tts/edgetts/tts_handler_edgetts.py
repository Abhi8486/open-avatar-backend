import io
import edge_tts
import os
import re
import time
from typing import Dict, Optional, cast
import librosa
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from abc import ABC
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.data_models.chat_signal import ChatSignal, ChatSignalType, ChatSignalSourceType
from engine_utils.directory_info import DirectoryInfo
import concurrent.futures

class TTSConfig(HandlerBaseConfigModel, BaseModel):
    ref_audio_path: str = Field(default=None)
    ref_audio_text: str = Field(default=None)
    voice: str = Field(default=None)
    sample_rate: int = Field(default=24000)


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config = None
        self.local_session_id = 0
        self.input_text = ''
        self.dump_audio = False
        self.audio_dump_file = None


class HandlerTTS(HandlerBase, ABC):
    def __init__(self):
        super().__init__()

        self.ref_audio_path = None
        self.ref_audio_text = None
        self.voice = None
        self.ref_audio_buffer = None
        self.sample_rate = None
      

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        text_def = DataBundleDefinition().add_entry(DataBundleEntry.create_string_entry("text_data"))
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            ),
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
                definition=text_def,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
       config = cast(TTSConfig, handler_config)
       self.voice = config.voice
       self.sample_rate = config.sample_rate
       self.ref_audio_path = config.ref_audio_path
       self.ref_audio_text = config.ref_audio_text


    def create_context(self, session_context, handler_config=None):
        if not isinstance(handler_config, TTSConfig):
            handler_config = TTSConfig()
        context = TTSContext(session_context.session_info.session_id)
        context.input_text = ''
        if context.dump_audio:
            dump_file_path = os.path.join(DirectoryInfo.get_project_dir(), 'temp',
                                            f"dump_avatar_audio_{context.session_id}_{time.localtime().tm_hour}_{time.localtime().tm_min}.pcm")
            context.audio_dump_file = open(dump_file_path, "wb")
        return context
    
    def start_context(self, session_context, context: HandlerContext):
        context = cast(TTSContext, context)
        edge_tts.Communicate(text="Test audio startup", voice=self.voice)

    def filter_text(self, text):
        pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # Match characters outside allowed ranges
        filtered_text = re.sub(pattern, "", text)
        return filtered_text

    def _generate_audio_with_timeout(self, sentence: str) -> Optional[bytes]:
        def fetch_audio():
            communicate = edge_tts.Communicate(sentence, self.voice)
            data = b''
            for chunk in communicate.stream_sync():
                if chunk['type'] == 'audio':
                    data += chunk['data']
            return data
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(fetch_audio)
            try:
                return future.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                logger.error(f"TTS generation timed out for sentence: {sentence}")
                return None
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                return None

    def _handle_tts_error(self, context: TTSContext, output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context.input_text = ''
        # Stop AvatarMusetalk and others
        context.emit_signal(ChatSignal(
            type=ChatSignalType.STREAM_CANCEL,
            source_type=ChatSignalSourceType.HANDLER,
            source_name="EdgeTTS"
        ))
        text_def = output_definitions.get(ChatDataType.AVATAR_TEXT)
        if text_def:
            err_bundle = DataBundle(text_def.definition)
            err_bundle.set_main_data("\n\n(TTS Error: There is an issue, please try again.)")
            context.submit_data((ChatDataType.AVATAR_TEXT, err_bundle), finish_stream=True)
        # Finish audio stream gracefully
        audio_def = output_definitions.get(ChatDataType.AVATAR_AUDIO)
        if audio_def:
            output = DataBundle(audio_def.definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            context.submit_data((ChatDataType.AVATAR_AUDIO, output), finish_stream=True)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type == ChatDataType.AVATAR_TEXT:
            text = inputs.data.get_main_data()
        else:
            return
        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            context.input_text += self.filter_text(text)

        text_end = inputs.data.get_meta("avatar_text_end", False) or inputs.is_last_data
        if not text_end:
            sentences = re.split(r'(?<=[,.~!?，。！？])', context.input_text)
            if len(sentences) > 1:  # At least one complete sentence
                complete_sentences = sentences[:-1]  # Complete sentences
                context.input_text = sentences[-1]  # Remaining incomplete part

                # Process complete sentences
                for sentence in complete_sentences:
                    if len(sentence.strip()) < 1:
                        continue
                    logger.info('current sentence' + sentence)
                    
                    data = self._generate_audio_with_timeout(sentence)
                    if data is None:
                        self._handle_tts_error(context, output_definitions)
                        return
                    
                    output_audio = librosa.load(io.BytesIO(data), sr=None)[0]
                    output_audio = output_audio[np.newaxis, ...]
                    output = DataBundle(output_definition)
                    output.set_main_data(output_audio)
                    context.submit_data((ChatDataType.AVATAR_AUDIO, output))
        else:
            logger.info('last sentence' + context.input_text)
            if context.input_text is not None and len(context.input_text.strip()) > 0:
                    data = self._generate_audio_with_timeout(context.input_text)
                    if data is None:
                        self._handle_tts_error(context, output_definitions)
                        return

                    output_audio = librosa.load(io.BytesIO(data), sr=None)[0]
                    output_audio = output_audio[np.newaxis, ...]
                    output = DataBundle(output_definition)
                    output.set_main_data(output_audio)
                    context.submit_data((ChatDataType.AVATAR_AUDIO, output))
            context.input_text = ''
            output = DataBundle(output_definition)
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
            context.submit_data((ChatDataType.AVATAR_AUDIO, output), finish_stream=True)
            logger.info(f"speech end")

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        logger.info('destroy context')

