import enum
import math
import os
from abc import ABC
from typing import cast, Dict, Optional, Tuple

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.common.handler_base import HandlerBase, HandlerDetail, HandlerDataInfo, HandlerBaseInfo
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal, SignalFilterRule
from chat_engine.data_models.chat_signal_type import ChatSignalType, ChatSignalSourceType
from chat_engine.data_models.chat_stream import ChatStreamIdentity
from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry
from chat_engine.data_models.runtime_data.event_model import EventType
from engine_utils.audio_utils import AudioUtils, AutoGainControl, create_mel_agc
from engine_utils.general_slicer import SliceContext, slice_data


class SileroVADConfigModel(HandlerBaseConfigModel, BaseModel):
    speaking_threshold: float = Field(default=0.5)
    start_delay: int = Field(default=2048)
    end_delay: int = Field(default=5000)
    early_end_delay: int = Field(default=1500, description="Early end detection threshold (in samples), used to trigger first early_vad_end event")
    early_end_repeat_delay: int = Field(default=3200, description="Interval for repeating early_vad_end during continuous silence (in samples), 0 means no repeat")
    buffer_look_back: int = Field(default=1024)
    prestart_fallback_threshold: int = Field(default=512)
    speech_padding: int = Field(default=512)
    volume_threshold: float = Field(default=-40)
    # Reconnection configuration
    post_end_monitor_samples: int = Field(default=16000, description="POST_END monitoring period length (in samples), 16000 = 1 sec")
    reconnect_threshold_samples: int = Field(default=8000, description="Reconnection threshold (in samples); values smaller than this are considered false triggers")
    # POST_END energy threshold (dB), fallback detection alongside VAD model
    # When audio energy exceeds this threshold, voice activity is recognized even if VAD model misses it
    post_end_energy_threshold: float = Field(default=-35, description="POST_END energy detection threshold (dB); above this value is considered voice activity")


class SpeakingStatus(enum.Enum):
    PRE_START = enum.auto()
    START = enum.auto()
    END = enum.auto()
    POST_END = enum.auto()  # Monitoring period after speech end


class HumanAudioVADContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.config: SileroVADConfigModel = SileroVADConfigModel()
        self.speaking_status = SpeakingStatus.END

        self.clip_size = 512

        self.audio_history = []
        self.history_length_limit = 0

        self.speech_length: int = 0
        self.silence_length: int = 0

        self.shared_states = None
        
        # Input enabled state (controlled by CLIENT_PLAYBACK STREAM_BEGIN/END signals)
        # In simplex mode: disabled on playback STREAM_BEGIN, re-enabled on playback STREAM_END
        self.input_enabled: bool = True

        self.model_state: Optional[np.ndarray] = None
        self.slice_context: Optional[SliceContext] = None

        self.agc: Optional[AutoGainControl] = None

        self.peak_volume = -100
        self.current_db: float = -100  # Current audio energy level (dB), fallback detection for POST_END

        self.speech_id: int = 0

        # EOU collaboration state
        self.early_silence_length: int = 0  # Early silence counter
        self.early_vad_end_count: int = 0  # Count of early_vad_end events sent in current stream
        self.last_early_vad_end_at: int = 0  # early_silence_length value when last early_vad_end was sent
        self.current_stream_id: Optional[ChatStreamIdentity] = None  # Track current stream
        self.candidate_end_received: bool = False  # Candidate end signal received flag

        # Reconnection state
        self.post_end_counter: int = 0  # Monitoring period counter
        self.post_end_speech_counter: int = 0  # Accumulated speech samples during POST_END (not reset by silence)
        self.last_stream_end_time: Optional[int] = None  # Previous stream end timestamp (sample ID)
        self.previous_stream_audio: list = []  # Complete audio buffer for previous stream (list of np.ndarray)
        self.previous_stream_id: Optional[ChatStreamIdentity] = None  # Previous stream ID
        self.current_stream_audio: list = []  # Audio buffer for current stream (saved for potential reconnection)
        self.post_end_speech_audio: list = []  # Speech audio detected during POST_END (sent on reconnection)
        # Persistent accumulated audio buffer, all clips from speech start until POST_END truly ends
        self.accumulated_speech_audio: list = []

    def reset_model(self):
        if self.agc is not None:
            self.agc.reset()
        self.model_state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset(self):
        self.audio_history.clear()
        self.speech_length = 0
        self.silence_length = 0
        self.slice_context.flush()
        # Reset EOU collaboration state
        self.early_silence_length = 0
        self.early_vad_end_count = 0
        self.last_early_vad_end_at = 0
        self.current_stream_id = None
        self.candidate_end_received = False
        # Reset reconnection state (note: previous_stream_audio and previous_stream_id are not reset here
        # because they must be retained until monitoring ends or reconnection completes)
        self.post_end_counter = 0
        self.current_stream_audio.clear()

    def reset_reconnect_state(self):
        """Reset reconnection state, called after monitoring period ends or reconnection completes"""
        self.post_end_counter = 0
        self.post_end_speech_counter = 0
        self.last_stream_end_time = None
        self.previous_stream_audio.clear()
        self.previous_stream_id = None
        self.current_stream_audio.clear()
        self.post_end_speech_audio.clear()
        # Only clear persistent accumulated audio when POST_END truly ends
        self.accumulated_speech_audio.clear()

    def _update_status_on_pre_start(self, clip: np.ndarray, _timestamp: Optional[int] = None):
        if self.speech_length >= self.config.start_delay:
            head_sample_id = None
            self.speaking_status = SpeakingStatus.START
            # Reset early_vad_end state so every new stream can trigger early_vad_end
            self.early_vad_end_count = 0
            self.last_early_vad_end_at = 0
            self.early_silence_length = 0
            self.candidate_end_received = False
            sample_num_to_fetch = self.config.buffer_look_back + self.config.start_delay
            slice_num_to_fetch = math.ceil(sample_num_to_fetch / self.clip_size)
            audio_clips = []
            for history_entry in self.audio_history[-slice_num_to_fetch:]:
                history_clip, history_timestamp = history_entry
                if head_sample_id is None:
                    head_sample_id = history_timestamp
                audio_clips.append(history_clip)
            output_audio = np.concatenate(audio_clips, axis=0)
            output_audio = np.concatenate(
                [np.zeros(self.config.speech_padding, dtype=clip.dtype), output_audio], axis=0)
            self.speech_id += 1
            logger.info("Start of human speech")
            extra_args =  {
                "human_speech_start": True,
                "pre_padding": self.config.speech_padding,
                "speech_length_at_start": self.speech_length,
            }
            if head_sample_id is not None:
                extra_args["head_sample_id"] = head_sample_id
                logger.info(f"VAD pre_start to start got timestamp {head_sample_id}")
            return output_audio, extra_args
        else:
            extra_args = {}
            if self.silence_length > self.config.prestart_fallback_threshold:
                logger.info("Back to not started status")
                self.speaking_status = SpeakingStatus.END
                extra_args["back_to_end"] = True
            return None, extra_args

    def _update_status_on_start(self, clip: np.ndarray, timestamp: Optional[int] = None):
        # Check if end conditions are met: normal end_delay or candidate end signal received
        should_end = (self.silence_length >= self.config.end_delay or 
                      self.candidate_end_received)
        
        if should_end:
            # Enter POST_END state, ending current stream (allowing ASR to process in time)
            # Monitoring continues during POST_END; if new speech is detected, this stream is cancelled
            self.speaking_status = SpeakingStatus.POST_END
            self.post_end_counter = 0
            self.post_end_speech_counter = 0  # Reset accumulated speech counter
            self.last_stream_end_time = timestamp
            self.post_end_speech_audio.clear()  # Clear speech buffer during POST_END
            output_audio = np.concatenate(
                [clip, np.zeros(self.config.speech_padding, dtype=clip.dtype)], axis=0)
            if self.candidate_end_received:
                logger.info("End of human speech (confirmed by EOU), entering POST_END monitoring")
            else:
                logger.info("End of human speech, entering POST_END monitoring")
            extra_args = {
                "human_speech_end": True,
                "post_padding": self.config.speech_padding,
                "silence_length_at_end": self.silence_length,
                "eou_confirmed": self.candidate_end_received,
                "entering_post_end": True,
            }
            if timestamp is not None:
                extra_args["head_sample_id"] = timestamp
                logger.info(f"VAD start to post_end got timestamp {timestamp}")
            return output_audio, extra_args
        else:
            extra_args = {"head_sample_id": timestamp}
            # Check if early_vad_end event needs to be sent
            should_send_early_end = False
            
            if self.early_vad_end_count == 0:
                # First time: send when early_end_delay is reached
                if self.early_silence_length >= self.config.early_end_delay:
                    should_send_early_end = True
            elif self.config.early_end_repeat_delay > 0:
                # Subsequent: send every early_end_repeat_delay
                silence_since_last = self.early_silence_length - self.last_early_vad_end_at
                if silence_since_last >= self.config.early_end_repeat_delay:
                    should_send_early_end = True
            
            if should_send_early_end:
                extra_args["early_vad_end"] = True
                self.early_vad_end_count += 1
                self.last_early_vad_end_at = self.early_silence_length
                logger.info(f"Early VAD end #{self.early_vad_end_count} at silence_length={self.early_silence_length}")
            
            return clip, extra_args

    def _update_status_on_post_end(self, clip: np.ndarray, timestamp: Optional[int] = None):
        """POST_END state: monitoring period after speech end judgment
        
        Stream has ended, but audio listening continues:
        - If new speech is detected and time gap is below threshold, cancel ended stream and reconnect
        - If monitoring period times out, confirm end judgment was correct and clean up state
        
        Dual mechanism for speech detection:
        1. VAD model detection (speech_length > 0)
        2. Energy detection (current_db > post_end_energy_threshold) - fallback for VAD model
        """
        self.post_end_counter += self.clip_size
        
        # Periodically output POST_END monitoring status (every 0.5 sec)
        if self.post_end_counter % 8000 == 0:  # 8000 samples = 0.5s
            progress_pct = (self.post_end_counter / self.config.post_end_monitor_samples) * 100
            logger.info(f"POST_END monitoring: {self.post_end_counter}/{self.config.post_end_monitor_samples} samples "
                       f"({progress_pct:.1f}%), speech_counter={self.post_end_speech_counter}, "
                       f"speech_length={self.speech_length}, current_db={self.current_db:.1f}dB")
        
        # During POST_END, use dual mechanism to detect speech:
        # 1. VAD model detection
        # 2. Energy detection as fallback (when VAD model might miss speech)
        vad_detected = self.speech_length > 0
        energy_detected = self.current_db > self.config.post_end_energy_threshold
        
        if vad_detected or energy_detected:
            self.post_end_speech_counter += self.clip_size
            self.post_end_speech_audio.append(clip.copy())
            self.accumulated_speech_audio.append(clip.copy())  # Persistent accumulation
            detection_source = "VAD" if vad_detected else "energy"
            if not vad_detected and energy_detected:
                detection_source = f"energy({self.current_db:.1f}dB > {self.config.post_end_energy_threshold}dB)"
            logger.info(f"POST_END: speech detected by {detection_source}, accumulated {self.post_end_speech_counter} samples")
        
        # Use accumulated counter to detect new speech (instead of speech_length)
        if self.post_end_speech_counter >= self.config.start_delay:
            # Calculate time gap
            time_gap = timestamp - self.last_stream_end_time if (timestamp is not None and self.last_stream_end_time is not None) else 0
            
            if time_gap < self.config.reconnect_threshold_samples:
                # Time gap below threshold: previous end judgment was false, cancel and reconnect
                logger.info(f"Reconnection triggered! Time gap {time_gap} < threshold {self.config.reconnect_threshold_samples}")
                self.speaking_status = SpeakingStatus.START
                # Reset early_vad_end state
                self.early_vad_end_count = 0
                self.last_early_vad_end_at = 0
                self.early_silence_length = 0
                self.candidate_end_received = False
                extra_args = {
                    "reconnect_triggered": True,
                    "time_gap": time_gap,
                    "head_sample_id": timestamp,
                }
                return clip, extra_args
            else:
                # Time gap above threshold: new independent speech, no reconnection needed
                logger.info(f"New speech detected but time gap {time_gap} >= threshold, starting new stream normally")
                self.speaking_status = SpeakingStatus.PRE_START
                self.reset_reconnect_state()
                return None, {"new_speech_no_reconnect": True}
        
        # Check if monitoring period timed out (speech detection takes priority)
        if self.post_end_counter >= self.config.post_end_monitor_samples:
            logger.info(f"POST_END monitoring period ended, confirming end was correct "
                       f"(accumulated speech: {self.post_end_speech_counter} samples, "
                       f"threshold: {self.config.start_delay})")
            self.speaking_status = SpeakingStatus.END
            self.reset_reconnect_state()
            return None, {"post_end_timeout": True}
        
        return None, {}

    def _update_status_on_end(self, _clip: np.ndarray, _timestamp: Optional[int] = None):
        if self.speech_length > 0:
            logger.info("Pre start of new human speech")
            self.speaking_status = SpeakingStatus.PRE_START
        return None, {}

    def _append_to_history(self, clip: np.ndarray, timestamp: Optional[int] = None):
        self.audio_history.append((clip, timestamp))
        while 0 < self.history_length_limit < len(self.audio_history):
            self.audio_history.pop(0)
    def update_status(self, speech_prob: float, clip: np.ndarray,
                      timestamp: Optional[int]=None) -> Tuple[Optional[np.ndarray], Dict]:
        self._append_to_history(clip, timestamp)
        if speech_prob > self.config.speaking_threshold:
            self.speech_length += self.clip_size
            self.silence_length = 0
            self.early_silence_length = 0  # Reset early silence counter when speech is present
            # Reset early_vad_end count on speech, allowing next pause to trigger again
            self.early_vad_end_count = 0
            self.last_early_vad_end_at = 0
        else:
            self.silence_length += self.clip_size
            self.early_silence_length += self.clip_size  # Accumulate early silence counter
            self.speech_length = 0
        if self.speaking_status == SpeakingStatus.PRE_START:
            return self._update_status_on_pre_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.START:
            return self._update_status_on_start(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.POST_END:
            return self._update_status_on_post_end(clip, timestamp)
        elif self.speaking_status == SpeakingStatus.END:
            return self._update_status_on_end(clip, timestamp)
        else:
            raise ValueError("Invalid speaking status")


class HandlerAudioVAD(HandlerBase, ABC):
    def __init__(self):
        super().__init__()
        self.model = None
        self.weight_curve = AudioUtils.get_a_weighting_curve()

    def get_handler_info(self):
        return HandlerBaseInfo(
            config_model=SileroVADConfigModel
        )

    def load(self, engine_config: ChatEngineConfigModel, handler_config = None):
        import onnxruntime
        model_name = "silero_vad.onnx"
        model_path = os.path.join(self.handler_root, "silero_vad",
                                  "src", "silero_vad", "data",
                                  model_name)
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self.model = onnxruntime.InferenceSession(model_path,
                                                  providers=["CPUExecutionProvider"],
                                                  sess_options=options)

    def create_context(self, session_context: SessionContext, handler_config = None) -> HandlerContext:
        context = HumanAudioVADContext(session_context.session_info.session_id)
        context.shared_states = session_context.shared_states
        if isinstance(handler_config, SileroVADConfigModel):
            context.config = handler_config
        context.slice_context = SliceContext.create_numpy_slice_context(
            slice_size=context.clip_size,
            slice_axis=0,
        )
        context.history_length_limit = math.ceil((context.config.start_delay + context.config.buffer_look_back)
                                                 / context.clip_size)
        context.agc = create_mel_agc(
            target_level_db=-5.0,
            max_gain_db=30.0,
            min_gain_db=-30.0,
            attack_time_ms=5.0,
            release_time_ms=50.0,
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
        )
        context.reset_model()
        return context

    def start_context(self, session_context, handler_context):
        pass

    def warmup_context(self, session_context: SessionContext, handler_context: HandlerContext):
        context = cast(HumanAudioVADContext, handler_context)
        if context.agc is not None:
            context.agc.warmup()

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("human_audio", 1, 16000))
        return HandlerDetail(
            inputs = [
                HandlerDataInfo(type=ChatDataType.MIC_AUDIO)
            ],
            outputs = [
                HandlerDataInfo(type=ChatDataType.HUMAN_AUDIO, definition=definition)
            ],
            signal_filters=[
                # Listen for STREAM_END signal targeted at HUMAN_AUDIO (used to receive candidate end signals from EOU)
                SignalFilterRule(ChatSignalType.STREAM_END, None, ChatDataType.HUMAN_AUDIO),
                # Listen for CLIENT_PLAYBACK stream lifecycle: pause audio input on STREAM_BEGIN, resume on STREAM_END (simplex mode)
                SignalFilterRule(ChatSignalType.STREAM_BEGIN, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_END, None, ChatDataType.CLIENT_PLAYBACK),
                SignalFilterRule(ChatSignalType.STREAM_CANCEL, None, ChatDataType.CLIENT_PLAYBACK),
            ]
        )

    def _inference(self, context: HumanAudioVADContext, clip: np.ndarray, sr: int=16000):
        clip = clip.squeeze()
        if clip.ndim != 1:
            logger.warning("Input audio should be 1-dim array")
            return 0
        clip = np.expand_dims(clip, axis=0)
        inputs = {
            "input": clip,
            "sr": np.array([sr], dtype=np.int64),
            "state": context.model_state
        }
        prob, state = self.model.run(None, inputs)
        context.model_state = state
        return prob[0][0]

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        context = cast(HumanAudioVADContext, context)
        output_definition = output_definitions.get(ChatDataType.HUMAN_AUDIO).definition
        
        # In POST_END state, audio must continue to be processed for monitoring regardless of playback status
        if not context.input_enabled and context.speaking_status != SpeakingStatus.POST_END:
            return
        if inputs.type != ChatDataType.MIC_AUDIO:
            return

        audio = inputs.data.get_main_data()
        if audio is None:
            return
        audio_entry = inputs.data.get_main_definition_entry()
        sample_rate = audio_entry.sample_rate
        audio = audio.squeeze()

        if context.agc is not None:
            context.agc.update_gain(audio)

        timestamp = None
        if inputs.is_timestamp_valid():
            timestamp = inputs.timestamp

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32767

        context.slice_context.update_start_id(timestamp[0], force_update=False)

        for clip in slice_data(context.slice_context, audio):
            rms = AudioUtils.get_rms(clip, self.weight_curve)
            db = AudioUtils.rms_to_db(rms)
            context.current_db = db  # Save current energy for POST_END fallback detection
            if db > context.peak_volume:
                context.peak_volume = db
            head_sample_id = context.slice_context.get_last_slice_start_index()
            speech_prob = self._inference(context, clip)
            if (context.speaking_status in (SpeakingStatus.END, SpeakingStatus.POST_END)
                and db < context.config.volume_threshold):
                speech_prob = 0.0
            # logger.info(f"RMS: {rms}, CurrentDB: {db} dB, PeakVolume: {context.peak_volume} dB, "
            #             f"VAD prob {speech_prob:.2f}: {'='*int(speech_prob * 20)}")
            if context.peak_volume > -100:
                context.peak_volume -= 1.0
            audio_clip, extra_args = context.update_status(speech_prob, clip, timestamp=head_sample_id)
            # FIXME this is a hack to disable VAD after human speech end,
            #  but it should be handled by client or downstream handlers
            human_speech_end = extra_args.get("human_speech_end", False)
            back_to_end = extra_args.get("back_to_end", False)
            entering_post_end = extra_args.get("entering_post_end", False)
            reconnect_triggered = extra_args.get("reconnect_triggered", False)
            post_end_timeout = extra_args.get("post_end_timeout", False)
            timestamp = extra_args.get("head_sample_id", head_sample_id)
            speech_id = f"speech-{context.session_id}-{context.speech_id}"
            
            # Entering POST_END state: save current stream info for potential cancellation
            if entering_post_end:
                streamer = context.data_submitter.get_streamer(ChatDataType.HUMAN_AUDIO)
                if streamer and streamer.current_stream:
                    context.previous_stream_id = streamer.current_stream.identity
                # Copy current_stream_audio to previous_stream_audio
                context.previous_stream_audio = context.current_stream_audio.copy()
                context.current_stream_audio.clear()
                logger.info(f"Entering POST_END, buffered {len(context.previous_stream_audio)} audio clips for potential reconnection")
            
            # Reconnection triggered: cancel ended stream and send buffered audio
            if reconnect_triggered:
                self._handle_reconnection(context, output_definition, sample_rate, speech_id)
            
            # POST_END timeout: confirm end judgment was correct
            if post_end_timeout:
                logger.info("POST_END timeout, clearing reconnection buffers")
            
            if human_speech_end:
                # In simplex mode, VAD self-disables after speech end.
                # It will be re-enabled by CLIENT_PLAYBACK STREAM_END when avatar finishes playing.
                context.input_enabled = False
            if back_to_end or (human_speech_end and not entering_post_end):
                context.reset()
                context.reset_model()
            if audio_clip is not None:
                if context.agc is not None:
                    audio_clip = context.agc.apply_gain(audio_clip)
                
                # Save audio to buffer while in START state (for potential reconnection)
                if context.speaking_status == SpeakingStatus.START or entering_post_end:
                    context.current_stream_audio.append(audio_clip.copy())
                    context.accumulated_speech_audio.append(audio_clip.copy())  # Persistent accumulation
                
                output = DataBundle(output_definition)
                output.set_main_data(np.expand_dims(audio_clip, axis=0))
                for flag_name, flag_value in extra_args.items():
                    output.add_meta(flag_name, flag_value)
                
                # Add Event if early_vad_end detected
                if extra_args.get("early_vad_end", False):
                    output.add_event_by_type(EventType.EVT_EARLY_VAD_END)

                output_chat_data = ChatData(
                    type=ChatDataType.HUMAN_AUDIO,
                    data=output
                )
                if extra_args.get("human_speech_end", False):
                    output_chat_data.is_last_data = True
                if timestamp >= 0:
                    output_chat_data.timestamp = timestamp, sample_rate
                context.submit_data(output_chat_data, finish_stream=output_chat_data.is_last_data)
                
                # Save current stream ID for signal handling
                streamer = context.data_submitter.get_streamer(ChatDataType.HUMAN_AUDIO)
                if streamer and streamer.current_stream:
                    context.current_stream_id = streamer.current_stream.identity

    def _handle_reconnection(self, context: HumanAudioVADContext, output_definition: DataBundleDefinition,
                             sample_rate: int, speech_id: str):
        """Handle reconnection logic: cancel ended stream, send buffered audio to new stream"""
        logger.info(f"Handling reconnection, cancelling previous stream and sending buffered audio")
        
        # 1. Cancel ended stream (cascades cancellation to all downstream streams)
        if context.previous_stream_id:
            streamer = context.data_submitter.get_streamer(ChatDataType.HUMAN_AUDIO)
            if streamer:
                cancelled = streamer.cancel_stream(context.previous_stream_id)
                if cancelled:
                    logger.info(f"Cancelled stream: {context.previous_stream_id}")
                else:
                    logger.warning(f"Failed to cancel stream: {context.previous_stream_id}")
        
        # 2. Re-enable VAD (restore directly on reconnection without waiting for CLIENT_PLAYBACK STREAM_END)
        context.input_enabled = True
        
        # 3. Send persistent accumulated audio to new stream
        # Use accumulated_speech_audio to ensure all audio is sent even across multiple reconnections
        total_clips = len(context.accumulated_speech_audio)
        
        if context.accumulated_speech_audio:
            logger.info(f"Sending {total_clips} accumulated audio clips to new stream")
            for clip_index, buffered_clip in enumerate(context.accumulated_speech_audio):
                output = DataBundle(output_definition)
                output.set_main_data(np.expand_dims(buffered_clip, axis=0))
                output.add_meta("reconnected_audio", True)
                output.add_meta("reconnected_audio_index", clip_index)
                
                output_chat_data = ChatData(
                    type=ChatDataType.HUMAN_AUDIO,
                    data=output
                )
                context.submit_data(output_chat_data, finish_stream=False)
        
        logger.info(f"Total {total_clips} buffered clips sent to new stream")
        
        # 4. Clean up temporary state, but preserve accumulated_speech_audio (cleared only when POST_END truly ends)
        context.post_end_counter = 0
        context.post_end_speech_counter = 0
        context.last_stream_end_time = None
        context.previous_stream_audio.clear()
        context.previous_stream_id = None
        context.current_stream_audio.clear()
        context.post_end_speech_audio.clear()
        
        # 5. Reset EOU state
        context.early_vad_end_count = 0
        context.last_early_vad_end_at = 0
        context.candidate_end_received = False

    def on_signal(self, context: HandlerContext, signal: ChatSignal):
        """Handle signals: CLIENT_PLAYBACK lifecycle controls input switch, EOU candidate end signals"""
        context = cast(HumanAudioVADContext, context)

        is_playback_stream = (signal.related_stream is not None
                              and signal.related_stream.data_type == ChatDataType.CLIENT_PLAYBACK)
        if is_playback_stream and signal.type in (ChatSignalType.STREAM_END, ChatSignalType.STREAM_CANCEL):
            context.input_enabled = True
            logger.debug(f"VAD input enabled by CLIENT_PLAYBACK {signal.type.value} from {signal.source_name}")
        elif is_playback_stream and signal.type == ChatSignalType.STREAM_BEGIN:
            context.input_enabled = False
            logger.debug(f"VAD input disabled by CLIENT_PLAYBACK STREAM_BEGIN from {signal.source_name}")
        elif signal.type == ChatSignalType.STREAM_END and not is_playback_stream and signal.is_candidate:
            # Candidate end signal received from another handler
            if (signal.related_stream and 
                context.current_stream_id and
                signal.related_stream.key == context.current_stream_id.key):
                # Confirm signal is for current stream, mark candidate end received
                context.candidate_end_received = True
                logger.info(f"Received candidate STREAM_END signal from {signal.source_name} for stream {signal.related_stream}")

    def destroy_context(self, context: HandlerContext):
        pass
