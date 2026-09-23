"""
WebSocket Input Endpoint Session Delegate
Handles client input and text echo
"""
import asyncio
import base64
import binascii
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Literal, Any
from uuid import uuid4

import numpy as np
from loguru import logger
from starlette.websockets import WebSocket, WebSocketState

from chat_engine.common.client_handler_base import ClientSessionDelegate
from chat_engine.contexts.session_clock import SessionClock
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_signal import ChatSignal
from chat_engine.data_models.chat_signal_type import ChatSignalSourceType, ChatSignalType
from chat_engine.data_models.chat_stream import ChatStreamIdentity
from chat_engine.data_models.chat_stream_config import ChatStreamConfig
from chat_engine.data_models.engine_channel_type import EngineChannelType
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition

from .ws_binary_protocol import BinaryStreamAssembler, BinaryPacketSplitter
from .ws_message_protocol import (
    parse_message, serialize_message,
    AvatarSessionInitialized, EchoHumanText, EchoAvatarText,
    EchoAvatarAudio, AvatarHeartbeat, InterruptAccepted, InterruptAcceptedPayload, Error,
    InterruptNotification, InterruptNotificationPayload,
    MessageHeader, EchoTextPayload, EchoAvatarAudioPayload, ErrorPayload, ErrorCode,
    InitializeAvatarSession, SendHumanAudio, SendHumanVideo,
    SendHumanText, TriggerHeartbeat, Interrupt, EndSpeech,
    MotionDataMessage, MotionDataPayload, BinaryDataInfo, ChatSignalPayload, ChatSignalMessage, MessageType,
    AudioFormat
)
from .ws_opus_codec import (
    OpusEncoder, OpusDecoder,
    is_opus_available,
    OpusStreamHeader,
    DEFAULT_OPUS_FRAME_SIZE_MS
)
from chat_engine.data_models.runtime_data.motion_data import MotionDataSerializer


@dataclass
class PendingBinaryMeta:
    kind: str  # 'audio' or 'video'
    message: Union[SendHumanAudio, SendHumanVideo]


@dataclass
class ConnectionInfo:
    connection_id: int
    websocket: WebSocket
    role: Literal["primary", "listener"]
    quit: asyncio.Event = field(default_factory=asyncio.Event)
    last_heartbeat_time: float = field(default_factory=time.time)
    motion_welcome_sent: bool = False


class WsInputSessionDelegate(ClientSessionDelegate):

    @staticmethod
    def _extract_stream_metadata(chat_data: ChatData, excluded_keys: set) -> Optional[Dict[str, Any]]:
        """
        Unified method for extracting stream metadata.

        Extract metadata from ChatData metadata, excluding fields handled separately in payload.
        ChatData metadata includes stream inheritable metadata (automatically copied by stream_manager).

        Args:
            chat_data: ChatData object
            excluded_keys: Metadata keys to exclude (already handled separately in payload)

        Returns:
            Filtered metadata dictionary, returns None if no valid metadata
        """
        if not chat_data.data or not chat_data.data.metadata:
            return None

        filtered_meta = {
            k: v for k, v in chat_data.data.metadata.items()
            if k not in excluded_keys
        }

        return filtered_meta if filtered_meta else None

    @staticmethod
    def _get_stream_metadata_for_signal(stream_manager, stream_id: ChatStreamIdentity) -> Optional[Dict[str, Any]]:
        """
        Unified method for retrieving stream metadata for signals.

        Retrieves complete stream metadata (regular + inheritable) for signals like STREAM_BEGIN.

        Args:
            stream_manager: StreamManager instance
            stream_id: Stream identity

        Returns:
            Complete metadata dictionary of stream, returns None if unavailable
        """
        if stream_manager is None or stream_id is None:
            return None

        try:
            stream = stream_manager.find_stream(stream_id)
            if stream is None:
                return None

            # Combine regular and inheritable metadata (inheritable takes priority)
            metadata = stream.metadata  # This property is already combined
            return metadata if metadata else None
        except Exception as e:
            logger.warning(f"Failed to get stream metadata for signal: {e}")
            return None

    def _enrich_signal_payload(self, signal: ChatSignal, signal_payload: ChatSignalPayload):
        """
        Enrich a ChatSignalPayload with parent_stream_keys and stream metadata
        based on signal type. Consolidates the per-signal-type enrichment logic
        so that STREAM_BEGIN and STREAM_CANCEL are handled uniformly.
        """
        if signal.type == ChatSignalType.STREAM_BEGIN:
            ancestry = self.stream_manager.get_stream_ancestry(signal.related_stream)
            parent_streams = ancestry.get('parents', [])
            if parent_streams:
                parent_stream_keys = [
                    s.stream_key_str for s in parent_streams if s.stream_key_str
                ]
                if parent_stream_keys:
                    signal_payload.parent_stream_keys = parent_stream_keys

            stream_metadata = self._get_stream_metadata_for_signal(
                self.stream_manager, signal.related_stream
            )
            if stream_metadata:
                if signal_payload.signal_data is None:
                    signal_payload.signal_data = {}
                signal_payload.signal_data['stream_metadata'] = stream_metadata

        elif (signal.type == ChatSignalType.STREAM_CANCEL
              and signal.related_stream is not None
              and signal.related_stream.data_type == ChatDataType.CLIENT_PLAYBACK):
            current_stream = self.stream_manager.find_stream(signal.related_stream)
            if current_stream is not None:
                signal_payload.parent_stream_keys = [
                    s.stream_key_str for s in current_stream.ancestor_streams
                ]

    """
    WebSocket input endpoint session delegate
    Handles audio/video/text input and control signals from client
    """

    AVAILABLE_SUBSCRIPTIONS = {
        "human_text",
        "avatar_text",
        "avatar_audio",
        "motion_data",
    }

    def __init__(self, heartbeat_timeout: float = 30.0):
        self.session_id: Optional[str] = None  # Session ID
        self.clock: Optional[SessionClock] = None
        self.data_submitter = None
        self.shared_states = None
        self.signal_emitter = None
        self.session_history = None  # SessionHistory for tracking client playback state
        self.stream_manager = None  # StreamManager for querying stream ancestry

        self.signal_to_client_queue = asyncio.Queue()

        # Output queues - used for receiving engine processed data
        self.output_queues = {
            EngineChannelType.AUDIO: asyncio.Queue(),
            EngineChannelType.TEXT: asyncio.Queue(),
            EngineChannelType.MOTION_DATA: asyncio.Queue(),
        }

        # Input data definitions
        self.input_data_definitions: Dict[EngineChannelType, DataBundleDefinition] = {}

        # Modality mapping
        self.modality_mapping = {
            EngineChannelType.AUDIO: ChatDataType.MIC_AUDIO,
            EngineChannelType.VIDEO: ChatDataType.CAMERA_VIDEO,
            EngineChannelType.TEXT: ChatDataType.HUMAN_TEXT,
        }

        # Session state
        self.quit = asyncio.Event()
        self.initialized = False
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat_time = time.time()

        # Binary data assembler (audio/video)
        self.binary_stream_assembler = BinaryStreamAssembler()

        # Text accumulation buffer (for incremental text accumulation)
        self.text_buffer: Dict[str, str] = {}  # speech_id -> accumulated_text
        self.last_human_text: Optional[str] = None

        # Motion Data output support
        self.motion_data_serializer: Optional[MotionDataSerializer] = None
        self.motion_welcome_sent = False
        self.motion_welcome_payload: Optional[bytes] = None

        # WebSocket send lock (prevents concurrent send conflicts)
        self.websocket_send_lock = asyncio.Lock()

        # Client playback tracking: stream_key -> playback_info dict
        # Records when audio was sent to client for playback
        self._active_playback_stream_keys: Dict[str, Dict] = {}

        # Cancelled stream_keys: Set of stream_keys that have been cancelled
        # Used to prevent sending audio for cancelled streams
        self._cancelled_stream_keys: set = set()

        # Lifecycle-only streamer for CLIENT_PLAYBACK tracking (lazy-initialized)
        self._playback_streamer = None

        # Connection management
        self.connection_infos: Dict[int, ConnectionInfo] = {}
        self.connection_lock = asyncio.Lock()
        self.primary_connection_id: Optional[int] = None
        self.primary_tasks: List[asyncio.Task] = []

        # Subscription configuration
        self.subscriptions: set[str] = set(self.AVAILABLE_SUBSCRIPTIONS)

        # Audio format configuration
        self.audio_format: str = "PCM"  # PCM or OPUS
        self.audio_sample_rate: int = 16000  # Input audio sample rate
        self.audio_channels: int = 1
        self.opus_frame_size_ms: int = DEFAULT_OPUS_FRAME_SIZE_MS

        # Opus codec (lazy loaded)
        self._opus_encoder: Optional[OpusEncoder] = None
        self._opus_decoder: Optional[OpusDecoder] = None

    async def get_data(self, modality: EngineChannelType, timeout: Optional[float] = 0.1) -> Optional[ChatData]:
        """Retrieve data from output queue."""
        data_queue = self.output_queues.get(modality)
        if data_queue is None:
            return None

        if timeout is not None and timeout > 0:
            try:
                data = await asyncio.wait_for(data_queue.get(), timeout)
            except asyncio.TimeoutError:
                return None
        else:
            data = await data_queue.get()

        return data

    def put_data(self, modality: EngineChannelType, data: Union[np.ndarray, str],
                 timestamp: Optional[Tuple[int, int]] = None,
                 samplerate: Optional[int] = None,
                 loopback: bool = False,
                 speech_id: Optional[str] = None):
        """
        Submit data to engine for processing.

        Args:
            modality: Data modality
            data: Data content
            timestamp: Timestamp
            samplerate: Sample rate
            loopback: Loopback flag
            speech_id: Utterance turn ID
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        if self.data_submitter is None:
            logger.warning("data_submitter is None, cannot submit data")
            return

        definition = self.input_data_definitions.get(modality)
        chat_data_type = self.modality_mapping.get(modality)

        if chat_data_type is None or definition is None:
            logger.warning(f"No definition for modality {modality}")
            return

        data_bundle = DataBundle(definition)
        is_last_data = False
        if modality == EngineChannelType.AUDIO:
            # Audio data: Ensure shape is [1, N]
            # Note: Do not add any metadata (consistent with RTC, VAD handles automatically)
            if isinstance(data, np.ndarray):
                data_bundle.set_main_data(data.squeeze()[np.newaxis, ...])
        elif modality == EngineChannelType.VIDEO:
            # Video data: Add batch dimension
            if isinstance(data, np.ndarray):
                data_bundle.set_main_data(data[np.newaxis, ...])
        elif modality == EngineChannelType.TEXT:
            # Text data
            is_last_data = True
            data_bundle.set_main_data(data)
        else:
            return

        chat_data = ChatData(
            source="client",
            type=chat_data_type,
            data=data_bundle,
            timestamp=timestamp,
        )
        self.data_submitter.submit(chat_data, finish_stream=is_last_data)

        if loopback:
            self.output_queues[modality].put_nowait(chat_data)

    def get_timestamp(self) -> Tuple[int, int]:
        """Get current timestamp."""
        return self.clock.get_timestamp()

    def emit_signal(self, signal: ChatSignal):
        """Emit signal to engine."""
        if self.signal_emitter is not None:
            self.signal_emitter.emit(signal)
        else:
            logger.warning("signal_emitter is None, cannot emit signal")

    def _get_playback_streamer(self):
        """Lazily create the CLIENT_PLAYBACK lifecycle streamer."""
        if self._playback_streamer is None and self.stream_manager is not None:
            self._playback_streamer = self.stream_manager.create_lifecycle_streamer(
                data_type=ChatDataType.CLIENT_PLAYBACK,
                producer_name="ws_client",
                config=ChatStreamConfig(cancelable=False),
            )
        return self._playback_streamer

    def clear_data(self):
        """Clear all queues and states."""
        for data_queue in self.output_queues.values():
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except Exception:
                    pass
        self.text_buffer.clear()
        self.binary_stream_assembler.clear()
        self.motion_welcome_sent = False
        self.motion_welcome_payload = None
        self.connection_infos.clear()
        self.primary_connection_id = None
        self._active_playback_stream_keys.clear()
        self._cancelled_stream_keys.clear()

    # ========================================================================
    # Connection Management and Broadcasting
    # ========================================================================

    async def _register_connection(self, websocket: WebSocket) -> ConnectionInfo:
        async with self.connection_lock:
            role: Literal["primary", "listener"]
            if self.primary_connection_id is None:
                role = "primary"
            else:
                role = "listener"
            connection_id = id(websocket)
            info = ConnectionInfo(connection_id=connection_id, websocket=websocket, role=role)
            self.connection_infos[connection_id] = info
            if role == "primary":
                self.primary_connection_id = connection_id
        logger.info(f"Connection registered for session {self.session_id}, role={role}, id={connection_id}")
        return info

    async def _close_connection(self, connection_id: int, code: int = 1000, reason: str = ""):
        async with self.connection_lock:
            info = self.connection_infos.pop(connection_id, None)
            if info and self.primary_connection_id == connection_id:
                self.primary_connection_id = None
        if info is None:
            return
        info.quit.set()
        try:
            if info.websocket.client_state != WebSocketState.DISCONNECTED:
                await info.websocket.close(code=code, reason=reason or "connection closed")
        except Exception:
            pass
        logger.info(f"Connection closed for session {self.session_id}, role={info.role}, id={connection_id}")

    async def _close_all_connections(self):
        async with self.connection_lock:
            connection_ids = list(self.connection_infos.keys())
        for connection_id in connection_ids:
            await self._close_connection(connection_id)
        logger.info(f"All connections closed for session {self.session_id}")

    async def _get_connection_snapshot(self) -> List[ConnectionInfo]:
        async with self.connection_lock:
            return list(self.connection_infos.values())

    async def _broadcast_json(self, json_data: dict):
        targets = await self._get_connection_snapshot()
        stale: List[int] = []
        async with self.websocket_send_lock:
            for info in targets:
                try:
                    await info.websocket.send_json(json_data)
                except Exception as e:
                    logger.warning(f"Failed to send json to connection {info.connection_id}: {e}")
                    stale.append(info.connection_id)
        for connection_id in stale:
            await self._close_connection(connection_id)

    async def _broadcast_bytes(self, data: bytes):
        if not data:
            return
        targets = await self._get_connection_snapshot()
        stale: List[int] = []
        async with self.websocket_send_lock:
            for info in targets:
                try:
                    await info.websocket.send_bytes(data)
                except Exception as e:
                    logger.warning(f"Failed to send bytes to connection {info.connection_id}: {e}")
                    stale.append(info.connection_id)
        for connection_id in stale:
            await self._close_connection(connection_id)

    async def _send_bytes_to_connection(self, connection_id: int, data: bytes):
        if not data:
            return
        async with self.connection_lock:
            info = self.connection_infos.get(connection_id)
        if info is None:
            return
        try:
            async with self.websocket_send_lock:
                await info.websocket.send_bytes(data)
        except Exception as e:
            logger.warning(f"Failed to send bytes to connection {connection_id}: {e}")
            await self._close_connection(connection_id)

    async def _send_message_to_connection(self, connection_id: int, message):
        if not message:
            return
        json_data = serialize_message(message)
        async with self.connection_lock:
            info = self.connection_infos.get(connection_id)
        if info is None:
            return
        try:
            async with self.websocket_send_lock:
                await info.websocket.send_json(json_data)
        except Exception as e:
            logger.warning(f"Failed to send bytes to connection {connection_id}: {e}")
            await self._close_connection(connection_id)

    async def _broadcast_message(self, message):
        json_data = serialize_message(message)
        await self._broadcast_json(json_data)

    async def _maybe_send_motion_welcome_to_connection(self, info: ConnectionInfo):
        if self.motion_welcome_payload and not info.motion_welcome_sent:
            motion_welcome_message = MotionDataMessage(
                header=MessageHeader(name="MotionDataWelcome", request_id=str(uuid4())),
                payload=MotionDataPayload(
                    stream_key="-1",
                    motion_data=BinaryDataInfo(
                        binary_size=len(self.motion_welcome_payload),
                        segment_num=1
                    ),
                    end_of_speech=True
                )
            )
            await self._send_message_to_connection(info.connection_id, motion_welcome_message)
            await self._send_bytes_to_connection(info.connection_id, self.motion_welcome_payload)
            info.motion_welcome_sent = True

    # ========================================================================
    # WebSocket Message Handling
    # ========================================================================

    async def _send_message(self, websocket: WebSocket, message):
        """Send JSON message."""
        try:
            json_data = serialize_message(message)
            async with self.websocket_send_lock:
                await websocket.send_json(json_data)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    async def _send_error(self, websocket: WebSocket, request_id: str, code: str, message: str):
        """Send error message."""
        error_msg = Error(
            header=MessageHeader(name="Error", request_id=request_id),
            payload=ErrorPayload(code=code, message=message)
        )
        await self._send_message(websocket, error_msg)

    async def _handle_initialize_session(self, websocket: WebSocket, msg: InitializeAvatarSession,
                                         connection_info: ConnectionInfo):
        """Handle session initialization."""
        logger.info(f"Initialize session: session_id={self.session_id}")

        if connection_info.role != "primary":
            if not self.initialized:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.INVALID_SESSION,
                    "Primary connection not initialized yet"
                )
                return False
            response = AvatarSessionInitialized(
                header=MessageHeader(name="AvatarSessionInitialized", request_id=msg.header.request_id)
            )
            await self._send_message(websocket, response)
            await self._maybe_send_motion_welcome_to_connection(connection_info)
            return True

        if self.initialized:
            response = AvatarSessionInitialized(
                header=MessageHeader(name="AvatarSessionInitialized", request_id=msg.header.request_id)
            )
            await self._send_message(websocket, response)
            return True

        # Validate audio configuration
        audio_config = msg.payload.audio
        audio_format_upper = audio_config.format.upper()

        # Validate audio format
        if audio_format_upper not in ["PCM", "OPUS"]:
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.AUDIO_FORMAT_ERROR,
                f"Unsupported audio format: {audio_config.format}. Expected PCM or OPUS."
            )
            return False

        # Validate sample rate
        if audio_config.sample_rate not in [8000, 12000, 16000, 24000, 48000]:
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.AUDIO_FORMAT_ERROR,
                f"Unsupported sample rate: {audio_config.sample_rate}. Expected 8000, 12000, 16000, 24000, or 48000."
            )
            return False

        # Validate channel count
        if audio_config.channels not in [1, 2]:
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.AUDIO_FORMAT_ERROR,
                f"Unsupported channel count: {audio_config.channels}. Expected 1 or 2."
            )
            return False

        # If OPUS format is used, check if opuslib is available
        if audio_format_upper == "OPUS" and not is_opus_available():
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.AUDIO_FORMAT_ERROR,
                "OPUS format requested but opuslib is not available on server."
            )
            return False

        # Save audio configuration
        self.audio_format = audio_format_upper
        self.audio_sample_rate = audio_config.sample_rate
        self.audio_channels = audio_config.channels
        self.opus_frame_size_ms = audio_config.opus_frame_size_ms or DEFAULT_OPUS_FRAME_SIZE_MS

        # Initialize Opus codec
        if audio_format_upper == "OPUS":
            try:
                self._opus_decoder = OpusDecoder(
                    sample_rate=16000,  # Resample to 16kHz after decoding for VAD/ASR
                    channels=1
                )
                # Encoder uses 24kHz (TTS output sample rate)
                self._opus_encoder = OpusEncoder(
                    sample_rate=24000,
                    channels=1,
                    frame_size_ms=self.opus_frame_size_ms
                )
                logger.info(f"Opus codec initialized for session {self.session_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Opus codec: {e}")
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.AUDIO_FORMAT_ERROR,
                    f"Failed to initialize Opus codec: {str(e)}"
                )
                return False
        else:
            self._opus_encoder = None
            self._opus_decoder = None

        logger.info(
            f"Audio config: format={self.audio_format}, sample_rate={self.audio_sample_rate}, "
            f"channels={self.audio_channels}, opus_frame_size_ms={self.opus_frame_size_ms}"
        )

        requested_subscriptions = msg.payload.subscriptions
        if requested_subscriptions is None:
            self.subscriptions = set(self.AVAILABLE_SUBSCRIPTIONS)
        else:
            normalized = {
                str(item).lower()
                for item in requested_subscriptions
                if isinstance(item, str)
            }
            unknown = normalized.difference(self.AVAILABLE_SUBSCRIPTIONS)
            if unknown:
                logger.warning(f"Unknown subscriptions {unknown}, ignoring.")
            filtered = normalized.intersection(self.AVAILABLE_SUBSCRIPTIONS)
            if not filtered:
                filtered = set(self.AVAILABLE_SUBSCRIPTIONS)
            self.subscriptions = filtered
        logger.info(f"Session {self.session_id} subscriptions: {self.subscriptions}")

        self.initialized = True

        # Send initialization completed message
        response = AvatarSessionInitialized(
            header=MessageHeader(name="AvatarSessionInitialized", request_id=msg.header.request_id)
        )
        await self._send_message(websocket, response)
        await self._maybe_send_motion_welcome_to_connection(connection_info)
        return True

    async def _handle_audio_data(self, websocket: WebSocket, msg: SendHumanAudio, audio_bytes: bytes):
        """Handle audio data (supports PCM and OPUS formats)."""
        try:
            if not audio_bytes:
                raise ValueError("Audio payload is empty")

            # Determine audio format (prefer format specified in message, otherwise use session config)
            audio_format = (msg.payload.format or self.audio_format).upper()

            if audio_format == "OPUS":
                # Opus decoding
                if self._opus_decoder is None:
                    # Attempt to create temporary decoder
                    if not is_opus_available():
                        raise ValueError("OPUS format requested but opuslib is not available")
                    self._opus_decoder = OpusDecoder(sample_rate=16000, channels=1)

                audio_array = self._opus_decoder.decode(audio_bytes)
                logger.debug(
                    f"Opus decoded: {len(audio_bytes)} bytes -> {audio_array.size} samples, "
                    f"session={self.session_id}"
                )
            else:
                # PCM format
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            if audio_array.size == 0:
                raise ValueError("Audio payload could not be parsed into samples")

            self.put_data(
                EngineChannelType.AUDIO,
                audio_array
            )

            # logger.debug(f"Received audio stream: {audio_array.size} samples, format={audio_format}, session={self.session_id}")
        except Exception as e:
            logger.error(f"Failed to process audio data: {e}")
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.AUDIO_FORMAT_ERROR,
                f"Failed to process audio data: {str(e)}"
            )

    async def _process_send_human_audio(self, websocket: WebSocket, msg: SendHumanAudio):
        """Process audio upload instruction based on transport."""
        transport = (msg.payload.transport or "binary").lower()

        if transport == "binary":
            if msg.payload.binary_size is None or msg.payload.segment_num is None:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "binary_size and segment_num are required for binary transport"
                )
                return
            if msg.payload.segment_num <= 0:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "segment_num must be greater than 0"
                )
                return
            metadata = PendingBinaryMeta(kind="audio", message=msg)
            self.binary_stream_assembler.register(
                request_id=msg.header.request_id,
                expected_segments=msg.payload.segment_num,
                expected_size=msg.payload.binary_size,
                metadata=metadata
            )
        elif transport == "base64":
            if not msg.payload.data_base64:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "data_base64 is required for base64 transport"
                )
                return
            try:
                audio_bytes = base64.b64decode(msg.payload.data_base64, validate=True)
            except (binascii.Error, ValueError) as e:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    f"Failed to decode base64 audio data: {str(e)}"
                )
                return
            await self._handle_audio_data(websocket, msg, audio_bytes)
        else:
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.INVALID_MESSAGE,
                f"Unsupported transport '{msg.payload.transport}'"
            )

    async def _process_send_human_video(self, websocket: WebSocket, msg: SendHumanVideo):
        """Process video upload instruction based on transport."""
        transport = (msg.payload.transport or "binary").lower()

        if transport == "binary":
            if msg.payload.binary_size is None or msg.payload.segment_num is None:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "binary_size and segment_num are required for binary transport"
                )
                return
            if msg.payload.segment_num <= 0:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "segment_num must be greater than 0"
                )
                return
            metadata = PendingBinaryMeta(kind="video", message=msg)
            self.binary_stream_assembler.register(
                request_id=msg.header.request_id,
                expected_segments=msg.payload.segment_num,
                expected_size=msg.payload.binary_size,
                metadata=metadata
            )
        elif transport == "base64":
            if not msg.payload.data_base64:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    "data_base64 is required for base64 transport"
                )
                return
            try:
                video_bytes = base64.b64decode(msg.payload.data_base64, validate=True)
            except (binascii.Error, ValueError) as e:
                await self._send_error(
                    websocket,
                    msg.header.request_id,
                    ErrorCode.BINARY_DATA_ERROR,
                    f"Failed to decode base64 video data: {str(e)}"
                )
                return
            await self._handle_video_data(websocket, msg, video_bytes)
        else:
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.INVALID_MESSAGE,
                f"Unsupported transport '{msg.payload.transport}'"
            )

    async def _handle_video_data(self, websocket: WebSocket, msg: SendHumanVideo, binary_data: bytes):
        """
        Handle video data.

        Internal system uses BGR format (consistent with RTC CAMERA_VIDEO)
        - JPEG/PNG: Decode as BGR using cv2.imdecode
        """
        try:
            import cv2

            format_upper = msg.payload.format.upper()

            # Decode video frame
            if format_upper in ["JPEG", "JPG", "PNG"]:
                # Compressed format: Decode with OpenCV (outputs BGR)
                video_array = np.frombuffer(binary_data, dtype=np.uint8)
                video_array = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

                if video_array is None:
                    raise ValueError(f"Failed to decode {format_upper} image")

                # Validate dimensions
                h, w = video_array.shape[:2]
                if h != msg.payload.height or w != msg.payload.width:
                    logger.warning(
                        f"Video size mismatch: expected {msg.payload.width}x{msg.payload.height}, "
                        f"got {w}x{h}"
                    )
            else:
                raise ValueError(f"Unsupported video format: {msg.payload.format}")

            # Validate final shape
            if video_array.shape[2] != 3:
                raise ValueError(f"Invalid video shape: {video_array.shape}, expected 3 channels")

            # Submit to engine (internal system uses BGR format)
            self.put_data(
                EngineChannelType.VIDEO,
                video_array
            )

            logger.debug(
                f"Received video frame: {video_array.shape}, format={format_upper}, "
                f"session={self.session_id}"
            )

        except Exception as e:
            logger.error(f"Failed to process video data: {e}")
            await self._send_error(
                websocket,
                msg.header.request_id,
                ErrorCode.VIDEO_FORMAT_ERROR,
                f"Failed to process video data: {str(e)}"
            )

    def _handle_end_speech(self, msg: EndSpeech):
        """Handle EndSpeech message."""
        stream_key_str = msg.payload.stream_key
        logger.info(f"EndSpeech received for stream_key={stream_key_str}")

        # Find and remove from active playback tracking
        # stream_key_str should be the key in _active_playback_stream_keys
        playback_info = None
        if stream_key_str in self._active_playback_stream_keys:
            playback_info = self._active_playback_stream_keys[stream_key_str]
            self._active_playback_stream_keys.pop(stream_key_str, None)
        else:
            # This should not happen in normal operation
            for key, info in self._active_playback_stream_keys.items():
                if isinstance(info, dict) and info.get("stream_key") == stream_key_str:
                    playback_info = info
                    self._active_playback_stream_keys.pop(key, None)
                    logger.warning(
                        f"EndSpeech: Found playback_info by speech_id fallback for stream_key={stream_key_str}, key={key}")
                    break

            if playback_info is None:
                logger.warning(
                    f"EndSpeech: No playback_info found for stream_key={stream_key_str}, playback stream may not have been opened")

        # Extract stream_key from playback_info
        # stream_key should always equal stream_key_str in the new design
        stream_key = stream_key_str
        if playback_info and isinstance(playback_info, dict):
            stored_stream_key = playback_info.get("stream_key")
            if stored_stream_key is not None and stored_stream_key != stream_key_str:
                logger.warning(f"EndSpeech: stream_key mismatch: received={stream_key_str}, stored={stored_stream_key}")
                stream_key = stored_stream_key  # Use stored value if different
            elif stored_stream_key is None:
                logger.warning(
                    f"EndSpeech: stream_key is None in playback_info for stream_key={stream_key_str}, using received value")

        # Close the CLIENT_PLAYBACK lifecycle stream
        # This auto-emits STREAM_END signal and auto-records to SessionHistory
        streamer = self._get_playback_streamer()
        if streamer is not None:
            streamer.finish_current()
            logger.info(f"CLIENT_PLAYBACK stream closed: stream_key={stream_key}")

    def _ensure_motion_serializer(self):
        if self.motion_data_serializer is None:
            self.motion_data_serializer = MotionDataSerializer()
            self.motion_data_serializer.register_audio_data("avatar_audio")
            self.motion_data_serializer.register_data(
                "arkit_face",
                "arkit_face",
                "float32"
            )
            logger.info("Motion data serializer initialized")

    async def _send_motion_welcome(self, definition):
        self._ensure_motion_serializer()
        try:
            welcome_data = self.motion_data_serializer.serialize(definition)

            motion_welcome_message = MotionDataMessage(
                header=MessageHeader(name="MotionDataWelcome", request_id=str(uuid4())),
                payload=MotionDataPayload(
                    stream_key="-1",
                    motion_data=BinaryDataInfo(
                        binary_size=len(welcome_data),
                        segment_num=1
                    ),
                    end_of_speech=True
                )
            )
            await self._broadcast_message(motion_welcome_message)
            self.motion_welcome_payload = welcome_data
            async with self.connection_lock:
                for info in self.connection_infos.values():
                    info.motion_welcome_sent = True
            await self._broadcast_bytes(welcome_data)
            self.motion_welcome_sent = True
            logger.info(f"Motion welcome message broadcast, size={len(welcome_data)} bytes")
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
            raise

    async def _send_motion_data(self, chat_data: ChatData):
        try:
            self._ensure_motion_serializer()
            binary_data = self.motion_data_serializer.serialize(
                chat_data.data, start_of_stream=chat_data.is_first_data, end_of_stream=chat_data.is_last_data)

            stream_key_str = chat_data.stream_id.stream_key_str if chat_data.stream_id else None
            # Use stream_key_str as the value for backward compatibility
            end_of_speech = chat_data.is_last_data

            # Check if this stream has been cancelled - skip sending if so
            if stream_key_str in self._cancelled_stream_keys:
                logger.debug(f"Skipping cancelled avatar audio: stream_key={stream_key_str}")
                return

            # Track client playback via lifecycle-only CLIENT_PLAYBACK stream
            if stream_key_str not in self._active_playback_stream_keys:
                self._active_playback_stream_keys[stream_key_str] = {
                    "start_time": time.monotonic(),
                    "stream_key": stream_key_str,
                }

                # Open a CLIENT_PLAYBACK lifecycle stream derived from AVATAR_AUDIO stream
                # This auto-emits STREAM_BEGIN signal and auto-records to SessionHistory
                streamer = self._get_playback_streamer()
                if streamer is not None:
                    sources = [chat_data.stream_id] if chat_data.stream_id else []
                    streamer.open_stream(sources=sources, name=f"playback:{stream_key_str}")
                    logger.info(f"CLIENT_PLAYBACK stream opened: stream_key={stream_key_str}")

            request_id = str(uuid4())
            segments = BinaryPacketSplitter.split(
                request_id=request_id[:8],
                packet_type=0,
                data=binary_data,
                segment_size=BinaryPacketSplitter.MOTION_DATA_SEGMENT_SIZE
            )

            json_msg = MotionDataMessage(
                header=MessageHeader(name="MotionData", request_id=request_id),
                payload=MotionDataPayload(
                    stream_key=stream_key_str,
                    motion_data=BinaryDataInfo(
                        binary_size=len(binary_data),
                        segment_num=len(segments)
                    ),
                    end_of_speech=end_of_speech
                )
            )

            await self._broadcast_message(json_msg)
            for segment in segments:
                await self._broadcast_bytes(segment)

            logger.debug(
                f"Motion data sent: stream_key={stream_key_str}, size={len(binary_data)}, "
                f"segments={len(segments)}, end_of_speech={end_of_speech}"
            )
        except Exception as e:
            logger.error(f"Failed to send motion data: {e}")
            raise

    async def _send_avatar_audio(self, chat_data: ChatData):
        if "avatar_audio" not in self.subscriptions:
            return

        try:
            audio_array = chat_data.data.get_main_data()
            if audio_array is None:
                logger.debug("No avatar audio data available to send.")
                return

            audio_np = np.asarray(audio_array)
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)

            if audio_np.size == 0:
                logger.debug("Avatar audio array is empty.")
                return

            # Get audio metadata
            definition = chat_data.data.definition
            audio_entry = definition.get_main_entry() if definition is not None else None
            channels = 1
            sample_rate = 24000
            if audio_entry is not None:
                if audio_entry.shape and isinstance(audio_entry.shape[0], int):
                    channels = audio_entry.shape[0]
                if audio_entry.sample_rate:
                    sample_rate = audio_entry.sample_rate

            # Get AVATAR_AUDIO stream info using stream_key
            # stream_id should always be present for valid streams
            if chat_data.stream_id is None:
                logger.error("AVATAR_AUDIO data missing stream_id, skipping send. This should not happen for valid streams.")
                return

            stream_key_str = chat_data.stream_id.stream_key_str
            if stream_key_str is None:
                logger.error(
                    f"AVATAR_AUDIO stream_id exists but stream_key_str is None, stream_id={chat_data.stream_id}, skipping send.")
                return

            # Check if this stream has been cancelled - skip sending if so
            if stream_key_str in self._cancelled_stream_keys:
                logger.debug(f"Skipping cancelled avatar audio: stream_key={stream_key_str}")
                return
            end_of_speech = chat_data.is_last_data

            # Track client playback via lifecycle-only CLIENT_PLAYBACK stream
            if stream_key_str not in self._active_playback_stream_keys:
                self._active_playback_stream_keys[stream_key_str] = {
                    "start_time": time.monotonic(),
                    "stream_key": stream_key_str,
                }

                # Open a CLIENT_PLAYBACK lifecycle stream derived from AVATAR_AUDIO stream
                # This auto-emits STREAM_BEGIN signal and auto-records to SessionHistory
                streamer = self._get_playback_streamer()
                if streamer is not None:
                    sources = [chat_data.stream_id] if chat_data.stream_id else []
                    streamer.open_stream(sources=sources, name=f"playback:{stream_key_str}")
                    logger.info(f"CLIENT_PLAYBACK stream opened: stream_key={stream_key_str}")

            # Determine output format according to session config
            output_format = self.audio_format
            opus_frame_size_ms = None

            if output_format == "OPUS" and self._opus_encoder is not None:
                # Opus encoding
                try:
                    # Ensure data is int16
                    if audio_np.dtype != np.int16:
                        if np.issubdtype(audio_np.dtype, np.floating):
                            audio_np = np.clip(audio_np, -1.0, 1.0)
                            audio_np = (audio_np * 32767.0).astype(np.int16)
                        else:
                            audio_np = audio_np.astype(np.int16)

                    # Recreate encoder if encoder sample rate does not match audio
                    if self._opus_encoder.sample_rate != sample_rate:
                        self._opus_encoder = OpusEncoder(
                            sample_rate=sample_rate,
                            channels=channels,
                            frame_size_ms=self.opus_frame_size_ms
                        )

                    # Encode audio, flush residual buffer at end of audio stream
                    binary_data = self._opus_encoder.encode(audio_np, flush=end_of_speech)
                    opus_frame_size_ms = self.opus_frame_size_ms

                    logger.debug(
                        f"Opus encoded avatar audio: {audio_np.size} samples -> {len(binary_data)} bytes, "
                        f"flush={end_of_speech}"
                    )
                except Exception as e:
                    logger.warning(f"Opus encoding failed, falling back to PCM: {e}")
                    output_format = "PCM"
                    binary_data = self._prepare_pcm_audio(audio_np)
            else:
                # PCM format
                output_format = "PCM"
                binary_data = self._prepare_pcm_audio(audio_np)

            binary_size = len(binary_data)
            base64_data = base64.b64encode(binary_data).decode("ascii") if binary_size > 0 else ""

            # Collect additional metadata (unified handling)
            stream_metadata = self._extract_stream_metadata(
                chat_data,
                excluded_keys=set()
            )

            message = EchoAvatarAudio(
                header=MessageHeader(name="EchoAvatarAudio", request_id=str(uuid4())),
                payload=EchoAvatarAudioPayload(
                    stream_key=stream_key_str,
                    transport="base64",
                    binary_size=None,
                    segment_num=None,
                    format=output_format,
                    sample_rate=sample_rate,
                    channels=channels,
                    data_base64=base64_data if base64_data else None,
                    end_of_speech=bool(end_of_speech),
                    opus_frame_size_ms=opus_frame_size_ms,
                    metadata=stream_metadata,
                )
            )

            await self._broadcast_message(message)

            logger.debug(
                f"Echo avatar audio sent ({output_format}): stream_key={stream_key_str}"
                f"size={binary_size}, sample_rate={sample_rate}, channels={channels}, end={end_of_speech}"
            )
        except Exception as e:
            logger.error(f"Failed to send avatar audio: {e}")
            raise

    def _prepare_pcm_audio(self, audio_np: np.ndarray) -> bytes:
        """Convert audio data to PCM format bytes."""
        if audio_np.dtype != np.int16:
            if np.issubdtype(audio_np.dtype, np.floating):
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_np = (audio_np * 32767.0).astype(np.int16)
            else:
                audio_np = audio_np.astype(np.int16)
        return audio_np.tobytes()

    async def _handle_text_data(self, websocket: WebSocket, msg: SendHumanText):
        """
        Handle text data.

        Client can send incremental or full text, but internal system always submits full text mode to engine.
        - If mode="increment": Accumulate text until end_of_speech=True to send full text
        - If mode="full_text": Use text directly (ignore previous accumulation)
        """
        try:
            stream_key_str = msg.payload.stream_key

            # Process text based on mode
            if msg.payload.mode == "increment":
                # Incremental mode: Accumulate text
                if stream_key_str not in self.text_buffer:
                    self.text_buffer[stream_key_str] = ""
                self.text_buffer[stream_key_str] += msg.payload.text
                full_text = self.text_buffer[stream_key_str]
            else:
                # Full text mode: Use directly
                full_text = msg.payload.text
                self.text_buffer[stream_key_str] = full_text

            logger.debug(f"Received text (mode={msg.payload.mode}): {msg.payload.text[:50]}...")

            text_end = msg.payload.end_of_speech

            response = EchoHumanText(
                header=MessageHeader(name="EchoHumanText", request_id=str(uuid4())),
                payload=EchoTextPayload(
                    stream_key=stream_key_str,
                    mode="full_text",  # HUMAN_TEXT is always full text
                    text=full_text,
                    end_of_speech=text_end
                )
            )
            await self._broadcast_message(response)
            logger.debug(f"Echo human text (end={text_end}): {full_text[:50]}...")

            # Only submit to engine when end_of_speech=True
            if msg.payload.end_of_speech:
                # Submit full text to engine (full text mode)
                self.put_data(
                    EngineChannelType.TEXT,
                    full_text,
                )

                # Clear buffer
                if stream_key_str in self.text_buffer:
                    del self.text_buffer[stream_key_str]

                logger.info(f"Submitted full text to engine: {full_text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to process text data: {e}")

    async def _handle_heartbeat(self, websocket: WebSocket, msg: TriggerHeartbeat,
                                connection_info: ConnectionInfo):
        """Handle heartbeat."""
        connection_info.last_heartbeat_time = time.time()
        # Send heartbeat response
        response = AvatarHeartbeat(
            header=MessageHeader(name="AvatarHeartbeat", request_id=msg.header.request_id)
        )
        await self._send_message(websocket, response)

    async def _handle_interrupt(self, websocket: WebSocket, msg: Interrupt):
        """Handle interrupt signal."""
        logger.info(f"Interrupt signal received: session_id={self.session_id}")

        # Reset Opus encoder, clear residual buffer
        if self._opus_encoder is not None:
            self._opus_encoder.reset()
            logger.debug("Opus encoder reset due to interrupt")

        # Send interrupt signal to engine
        signal = ChatSignal(
            source_type=ChatSignalSourceType.CLIENT,
            type=ChatSignalType.INTERRUPT,
            source_name="ws_client",
        )
        self.emit_signal(signal)

        # Send interrupt accepted message (no target info since client initiated)
        response = InterruptAccepted(
            header=MessageHeader(name="InterruptAccepted", request_id=msg.header.request_id),
            payload=None  # Client-initiated interrupt doesn't have specific target info
        )
        await self._send_message(websocket, response)

    async def _handle_server_interrupt_signal(self, signal: ChatSignal):
        """
        Handle server-initiated INTERRUPT semantic signal.

        INTERRUPT is a pure semantic notification ("system decided to interrupt"), carrying no specific stream reference.
        Actual stream cancellation and browser notifications are handled by STREAM_CANCEL(AVATAR_AUDIO) handler.
        Only instant interrupt-specific cleanup (e.g. opus encoder reset) is performed here.
        """
        signal_data = signal.signal_data or {}
        reason = signal_data.get("reason", "semantic_interrupt")
        logger.info(f"Server interrupt signal received: reason={reason}")

        if self._opus_encoder is not None:
            self._opus_encoder.reset()
            logger.debug("Opus encoder reset due to server interrupt")

    async def _handle_avatar_audio_cancel_signal(self, signal: ChatSignal):
        """
        Handle AVATAR_AUDIO stream cancel signal (emitted by stream system cancel()).

        This is the single entry point for browser notifications in the interrupt chain:
        - cancel_streams_by_type(CLIENT_PLAYBACK) -> cancel_stream_chain -> AVATAR_AUDIO.cancel()
        - AVATAR_AUDIO.cancel() emits STREAM_CANCEL signal -> arrives here
        - Sends InterruptNotification here to notify browser to stop playback

        CLIENT_PLAYBACK stream is automatically cascaded and cancelled by engine forward_cancel_signal.
        """
        if signal.related_stream is None:
            logger.warning("STREAM_CANCEL signal missing related_stream")
            return

        stream_key_str = signal.related_stream.stream_key_str if signal.related_stream else None
        if not stream_key_str:
            logger.debug("AVATAR_AUDIO cancel signal: could not identify stream_key, skipping")
            return

        # Idempotent: skip if already processed
        if stream_key_str in self._cancelled_stream_keys:
            logger.debug(f"AVATAR_AUDIO cancel: stream_key={stream_key_str} already cancelled, skipping")
            return

        logger.info(f"Processing AVATAR_AUDIO cancel signal: stream_key={stream_key_str}")
        self._cancelled_stream_keys.add(stream_key_str)

        playback_active = stream_key_str in self._active_playback_stream_keys
        if playback_active:
            self._active_playback_stream_keys.pop(stream_key_str, None)

            notification = InterruptNotification(
                header=MessageHeader(name="InterruptNotification", request_id=str(uuid4())),
                payload=InterruptNotificationPayload(
                    target_stream_id=stream_key_str,
                    stream_key=stream_key_str,
                    reason="stream_cancelled",
                    interrupted_at=time.time(),
                )
            )
            await self._broadcast_message(notification)
            logger.info(f"InterruptNotification sent for cancelled AVATAR_AUDIO: stream_key={stream_key_str}")

            if self._opus_encoder is not None:
                self._opus_encoder.reset()

    # ========================================================================
    # WebSocket Tasks
    # ========================================================================

    async def _ws_input_task(self, connection_info: ConnectionInfo):
        """Receive task - handles client messages."""
        websocket = connection_info.websocket
        role = connection_info.role
        logger.info(f"Input task started for session {self.session_id}, role={role}")

        while not connection_info.quit.is_set() and not self.quit.is_set():
            try:
                # Receive message (may be JSON or binary)
                raw_msg = await asyncio.wait_for(websocket.receive(), timeout=0.1)

                # Process JSON message
                if "text" in raw_msg:
                    # Update heartbeat time (any message indicates client activity)
                    connection_info.last_heartbeat_time = time.time()

                    json_data = json.loads(raw_msg["text"])
                    msg = parse_message(json_data)

                    if msg is None:
                        logger.warning(f"Failed to parse message: {json_data}")
                        continue

                    # Route message
                    if isinstance(msg, InitializeAvatarSession):
                        success = await self._handle_initialize_session(websocket, msg, connection_info)
                        if role == "primary" and not success:
                            break
                    elif isinstance(msg, TriggerHeartbeat):
                        await self._handle_heartbeat(websocket, msg, connection_info)
                    elif role != "primary":
                        await self._send_error(
                            websocket,
                            msg.header.request_id if msg else str(uuid4()),
                            ErrorCode.INVALID_MESSAGE,
                            "Only primary connection can send this message"
                        )
                    elif isinstance(msg, SendHumanAudio):
                        await self._process_send_human_audio(websocket, msg)
                    elif isinstance(msg, SendHumanVideo):
                        await self._process_send_human_video(websocket, msg)
                    elif isinstance(msg, SendHumanText):
                        await self._handle_text_data(websocket, msg)
                    elif isinstance(msg, Interrupt):
                        await self._handle_interrupt(websocket, msg)
                    elif isinstance(msg, EndSpeech):
                        self._handle_end_speech(msg)

                # Process binary message
                elif "bytes" in raw_msg:
                    # Update heartbeat time (any message indicates client activity)
                    connection_info.last_heartbeat_time = time.time()

                    if role != "primary":
                        logger.warning("Listener connection received unexpected binary payload, ignoring.")
                        continue

                    binary_data = raw_msg["bytes"]

                    result = self.binary_stream_assembler.append(binary_data)

                    if result is not None:
                        state, complete_data = result
                        metadata = state.metadata
                        if isinstance(metadata, PendingBinaryMeta):
                            if metadata.kind == "audio":
                                await self._handle_audio_data(websocket, metadata.message, complete_data)
                            elif metadata.kind == "video":
                                await self._handle_video_data(websocket, metadata.message, complete_data)
                            else:
                                logger.warning(f"Unknown binary metadata kind: {metadata.kind}")
                        else:
                            logger.warning("Missing metadata for completed binary transfer")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in input task for session {self.session_id}: {e}")
                break

        connection_info.quit.set()
        logger.info(f"Input task ended for session {self.session_id}, role={role}")

    async def _ws_text_output_task(self):
        """
        Send task - sends text echo.

        Handles two text outputs:
        - HUMAN_TEXT: Engine returns full text (human_text_end=True), sent in full text mode
        - AVATAR_TEXT: Engine returns incremental text (avatar_text_end=False/True), sent in incremental mode
        """
        logger.info(f"Text output task started for session {self.session_id}")

        while not self.quit.is_set():
            try:
                # Get text data (ASR or LLM output)
                chat_data = await asyncio.wait_for(
                    self.get_data(EngineChannelType.TEXT),
                    timeout=0.1
                )

                if chat_data is None:
                    continue

                # Extract text content and metadata
                text = chat_data.data.get_main_data()

                stream_key_str = chat_data.stream_id.stream_key_str if chat_data.stream_id else None

                # Send different echo messages based on type
                if chat_data.type == ChatDataType.HUMAN_TEXT:
                    if "human_text" not in self.subscriptions:
                        continue
                    # HUMAN_TEXT: Engine sends full text
                    # Get human_text_end flag (default True)
                    text_end = chat_data.is_last_data

                    # Collect additional metadata (unified handling)
                    stream_metadata = self._extract_stream_metadata(
                        chat_data,
                        excluded_keys={'human_text_end'}
                    )

                    response = EchoHumanText(
                        header=MessageHeader(name="EchoHumanText", request_id=str(uuid4())),
                        payload=EchoTextPayload(
                            stream_key=stream_key_str,
                            mode="full_text",  # HUMAN_TEXT is always full text
                            text=text,
                            end_of_speech=text_end,
                            metadata=stream_metadata,
                        )
                    )
                    await self._broadcast_message(response)
                    logger.debug(f"Echo human text (end={text_end}): {text[:50]}...")
                    self.last_human_text = text if not text_end else None

                elif chat_data.type == ChatDataType.AVATAR_TEXT:
                    if "avatar_text" not in self.subscriptions:
                        continue
                    # AVATAR_TEXT: Engine sends incremental text
                    # Get avatar_text_end flag (default False)
                    text_end = chat_data.is_last_data

                    # Collect additional metadata (unified handling)
                    stream_metadata = self._extract_stream_metadata(
                        chat_data,
                        excluded_keys={'avatar_text_end'}
                    )

                    response = EchoAvatarText(
                        header=MessageHeader(name="EchoAvatarText", request_id=str(uuid4())),
                        payload=EchoTextPayload(
                            stream_key=stream_key_str,
                            mode="increment",  # AVATAR_TEXT is always incremental
                            text=text,
                            end_of_speech=text_end,
                            metadata=stream_metadata,
                        )
                    )
                    await self._broadcast_message(response)
                    logger.debug(f"Echo avatar text (end={text_end}): {text[:50]}...")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in output task for session {self.session_id}: {e}")
                break

        logger.info(f"Text output task ended for session {self.session_id}")

    async def _ws_motion_output_task(self):
        """Output task - sends Motion Data."""
        logger.info(f"Motion output task started for session {self.session_id}")

        while not self.quit.is_set():
            try:
                chat_data = await asyncio.wait_for(
                    self.get_data(EngineChannelType.MOTION_DATA),
                    timeout=0.1
                )

                if chat_data is None:
                    continue

                if "motion_data" not in self.subscriptions:
                    self.motion_welcome_sent = False
                    continue

                if not self.motion_welcome_sent:
                    await self._send_motion_welcome(chat_data.data.definition)

                if chat_data.type == ChatDataType.AVATAR_MOTION_DATA:
                    await self._send_motion_data(chat_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in motion output task for session {self.session_id}: {e}")
                break

        logger.info(f"Motion output task ended for session {self.session_id}")

    async def _heartbeat_monitor_task(self, connection_info: ConnectionInfo):
        """Heartbeat monitor task."""
        logger.info(f"Heartbeat monitor started for session {self.session_id}, role={connection_info.role}")

        while not connection_info.quit.is_set() and not self.quit.is_set():
            try:
                await asyncio.sleep(1.0)

                # Check heartbeat timeout
                elapsed = time.time() - connection_info.last_heartbeat_time
                if elapsed > self.heartbeat_timeout:
                    logger.warning(
                        f"Heartbeat timeout for session {self.session_id} role={connection_info.role}: {elapsed:.1f}s"
                    )
                    await self._send_error(
                        connection_info.websocket,
                        str(uuid4()),
                        ErrorCode.HEARTBEAT_TIMEOUT,
                        f"Heartbeat timeout after {elapsed:.1f} seconds"
                    )
                    connection_info.quit.set()

                    break

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                break

        logger.info("Heartbeat monitor ended")

    async def _ws_audio_output_task(self):
        """Output task - sends Avatar Audio."""
        logger.info(f"Audio output task started for session {self.session_id}")

        while not self.quit.is_set():
            try:
                chat_data = await asyncio.wait_for(
                    self.get_data(EngineChannelType.AUDIO),
                    timeout=0.1
                )

                if chat_data is None:
                    continue

                if chat_data.type != ChatDataType.AVATAR_AUDIO:
                    logger.debug(f"Skip non-avatar audio data type: {chat_data.type}")
                    continue

                await self._send_avatar_audio(chat_data)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in audio output task for session {self.session_id}: {e}")
                break

        logger.info(f"Audio output task ended for session {self.session_id}")

    async def _ws_signal_output_task(self):
        """
        Send signals - sends server signals to client.
        """
        logger.info(f"Signal output task started for session {self.session_id}")

        while not self.quit.is_set():
            try:
                signal: ChatSignal = await asyncio.wait_for(
                    self.signal_to_client_queue.get(),
                    timeout=0.1
                )

                if signal is None:
                    continue

                # Special handling for INTERRUPT signals from server (SemanticTurnDetector)
                # Send InterruptNotification to client for precise audio stop
                if signal.type == ChatSignalType.INTERRUPT:
                    if signal.source_type == ChatSignalSourceType.HANDLER:
                        await self._handle_server_interrupt_signal(signal)
                    # CLIENT-sourced INTERRUPTs originated from this WS client;
                    # forwarding them back would cause the client to re-send
                    # Interrupt messages, creating an infinite loop.
                    continue

                # Special handling for STREAM_CANCEL signals on AVATAR_AUDIO
                # Mark the speech as cancelled and notify client to stop playback
                if (signal.type == ChatSignalType.STREAM_CANCEL
                    and signal.related_stream is not None
                        and signal.related_stream.data_type == ChatDataType.AVATAR_AUDIO):
                    await self._handle_avatar_audio_cancel_signal(signal)
                    continue

                # Forward other signals as generic ChatSignalMessage
                signal_payload = ChatSignalPayload(
                    type=signal.type,
                    source_type=signal.source_type,
                    signal_data=signal.signal_data,
                )
                if signal.related_stream is not None:
                    signal_payload.stream_type = signal.related_stream.data_type.value
                    signal_payload.stream_producer = signal.related_stream.producer_name
                    signal_payload.stream_key = signal.related_stream.stream_key_str

                    if self.stream_manager is not None:
                        try:
                            self._enrich_signal_payload(signal, signal_payload)
                        except Exception as e:
                            logger.warning(f"Failed to enrich signal payload for {signal.type}: {e}")

                message = ChatSignalMessage(
                    header=MessageHeader(name=MessageType.CHAT_SIGNAL, request_id=str(uuid4())),
                    payload=signal_payload
                )
                await self._broadcast_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in signal output task for session {self.session_id}: {e}")
                break
        logger.info(f"Signal output task ended for session {self.session_id}")

    async def serve_websocket(self, websocket: WebSocket) -> bool:
        """Serve WebSocket connection, returning whether the entire Session should be destroyed."""
        info = await self._register_connection(websocket)
        logger.info(f"Serving WebSocket connection role={info.role}, session={self.session_id}")

        if info.role == "primary":
            return await self._serve_primary_connection(info)
        else:
            await self._serve_listener_connection(info)
            return False

    async def _serve_primary_connection(self, info: ConnectionInfo) -> bool:
        """Start primary connection tasks, returns True upon completion to destroy Session."""

        # Reset state
        self.quit.clear()
        self.motion_welcome_sent = False
        self.motion_welcome_payload = None
        self.binary_stream_assembler.clear()
        self.subscriptions = set(self.AVAILABLE_SUBSCRIPTIONS)

        # Reset audio configuration
        self.audio_format = "PCM"
        self.audio_sample_rate = 16000
        self.audio_channels = 1
        self._opus_encoder = None
        self._opus_decoder = None

        self.primary_tasks = [
            asyncio.create_task(self._ws_input_task(info)),
            asyncio.create_task(self._ws_text_output_task()),
            asyncio.create_task(self._ws_audio_output_task()),
            asyncio.create_task(self._ws_motion_output_task()),
            asyncio.create_task(self._heartbeat_monitor_task(info)),
            asyncio.create_task(self._ws_signal_output_task()),
        ]

        try:
            # Wait for any task to finish
            done, pending = await asyncio.wait(
                self.primary_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel other tasks
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        finally:
            for task in self.primary_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self.primary_tasks, return_exceptions=True)
            self.primary_tasks = []
            self.quit.set()
            await self._close_all_connections()
            self.motion_welcome_sent = False
            self.motion_welcome_payload = None
            self.binary_stream_assembler.clear()
            logger.info(f"Primary connection closed, session={self.session_id}")
        return True

    async def _serve_listener_connection(self, info: ConnectionInfo):
        """Start listener connection tasks."""
        tasks = [
            asyncio.create_task(self._ws_input_task(info)),
            asyncio.create_task(self._heartbeat_monitor_task(info))
        ]

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._close_connection(info.connection_id)
            logger.info(f"Listener connection closed, session={self.session_id}")
