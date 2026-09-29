from typing import Dict, Optional, cast
from loguru import logger
import numpy as np
import os
import io
import requests
import scipy.io.wavfile as wavfile
from pydantic import BaseModel, Field
from abc import ABC

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.contexts.session_context import SessionContext

from engine_utils.general_slicer import SliceContext, slice_data


class ASRSarvamConfig(HandlerBaseConfigModel, BaseModel):
    api_key: Optional[str] = Field(default=None)


class ASRSarvamContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.output_audios = []
        self.audio_slice_context = SliceContext.create_numpy_slice_context(
            slice_size=16000,
            slice_axis=0,
        )


class HandlerASRSarvam(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("SARVAM_API_KEY", "")

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="ASR_Sarvam",
            config_model=ASRSarvamConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(
            inputs=inputs, outputs=outputs,
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        if isinstance(handler_config, ASRSarvamConfig) and handler_config.api_key:
            self.api_key = handler_config.api_key
        logger.info("Loaded Sarvam ASR Handler.")

    def create_context(self, session_context, handler_config=None):
        return ASRSarvamContext(session_context.session_info.session_id)

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):

        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRSarvamContext, context)
        
        if inputs.type == ChatDataType.HUMAN_AUDIO:
            audio = inputs.data.get_main_data()
        else:
            return

        if audio is not None:
            audio = audio.squeeze()
            for audio_segment in slice_data(context.audio_slice_context, audio):
                if audio_segment is None or audio_segment.shape[0] == 0:
                    continue
                context.output_audios.append(audio_segment)

        if not inputs.is_last_data:
            return

        # VAD detected speech end, process accumulated audio
        remainder_audio = context.audio_slice_context.flush()
        if remainder_audio is not None:
            if remainder_audio.shape[0] < context.audio_slice_context.slice_size:
                remainder_audio = np.concatenate(
                    [remainder_audio,
                     np.zeros(shape=(context.audio_slice_context.slice_size - remainder_audio.shape[0]))])
            context.output_audios.append(remainder_audio)
        
        if len(context.output_audios) == 0:
            return

        output_audio = np.concatenate(context.output_audios)
        context.output_audios.clear()

        # Convert to wav bytes
        wav_io = io.BytesIO()
        wavfile.write(wav_io, 16000, output_audio.astype(np.int16))
        wav_io.seek(0)

        # Call Sarvam AI API
        logger.info("Sending audio to Sarvam AI STT...")
        headers = {
            "api-subscription-key": self.api_key
        }
        data = {
            "model": "saaras:v4"
        }
        files = {
            "file": ("audio.wav", wav_io, "audio/wav")
        }

        try:
            response = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=data, files=files)
            if response.status_code == 200:
                result = response.json()
                transcript = result.get('transcript', '').strip()
                if transcript:
                    logger.info(f"Sarvam ASR Transcript: {transcript}")
                    output = DataBundle(output_definition)
                    output.set_main_data(transcript)
                    context.submit_data(output, finish_stream=True)
                else:
                    logger.warning("Sarvam ASR returned empty transcript.")
            else:
                logger.error(f"Sarvam AI Error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Failed to call Sarvam AI STT API: {e}")

    def destroy_context(self, context: HandlerContext):
        pass
