"""
WebSocket message protocol definition
Defines Pydantic models for all JSON messages
"""
import time
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum
from loguru import logger


# ============================================================================
# Base Structures
# ============================================================================

class MessageType(str, Enum):
    """Message type enumeration"""
    # Input Port - Client Messages
    INITIALIZE_AVATAR_SESSION = "InitializeAvatarSession"
    SEND_HUMAN_AUDIO = "SendHumanAudio"
    SEND_HUMAN_VIDEO = "SendHumanVideo"
    SEND_HUMAN_TEXT = "SendHumanText"
    TRIGGER_HEARTBEAT = "TriggerHeartbeat"
    INTERRUPT = "Interrupt"
    
    # Input Port - Server Messages
    AVATAR_SESSION_INITIALIZED = "AvatarSessionInitialized"
    ECHO_HUMAN_TEXT = "EchoHumanText"
    ECHO_AVATAR_TEXT = "EchoAvatarText"
    ECHO_AVATAR_AUDIO = "EchoAvatarAudio"
    AVATAR_HEARTBEAT = "AvatarHeartbeat"
    INTERRUPT_ACCEPTED = "InterruptAccepted"
    INTERRUPT_NOTIFICATION = "InterruptNotification"  # Server-initiated interrupt (for non-MotionData mode)
    CHAT_SIGNAL = "ChatSignal"
    ERROR = "Error"
    
    # Output Port - Server Messages
    MOTION_DATA = "MotionData"
    MOTION_DATA_WELCOME = "MotionDataWelcome"
    
    # Input Port - Renderer Messages
    END_SPEECH = "EndSpeech"

class MessageHeader(BaseModel):
    """Message Header"""
    name: MessageType
    request_id: str = Field(..., description="Request ID")


class BaseMessage(BaseModel):
    """Base Message Structure"""
    header: MessageHeader
    payload: Optional[Dict[str, Any]] = Field(default=None)


# ============================================================================
# Input Port - Client -> Server Messages
# ============================================================================

class AudioFormat(str, Enum):
    """Audio Format Enumeration"""
    PCM = "PCM"
    OPUS = "OPUS"


class AudioConfig(BaseModel):
    """Audio Configuration"""
    format: str = Field(default="PCM", description="Audio format: PCM or OPUS")
    sample_rate: int = Field(default=16000, description="Sample rate")
    channels: int = Field(default=1, description="Number of channels")
    
    # Opus specific configuration
    opus_frame_size_ms: Optional[int] = Field(
        default=20, 
        description="Opus frame duration in ms (valid only when format=OPUS)"
    )


class InitializeAvatarSessionPayload(BaseModel):
    """Initialize session payload"""
    audio: AudioConfig
    subscriptions: Optional[List[str]] = Field(default=None, description="Subscribed downstream contents")


class InitializeAvatarSession(BaseModel):
    """Initialize Avatar Session"""
    header: MessageHeader
    payload: InitializeAvatarSessionPayload


class BinaryDataInfo(BaseModel):
    """Binary Data Information"""
    binary_size: int = Field(..., description="Total binary size")
    segment_num: int = Field(..., description="Number of segments")


class SendHumanAudioPayload(BaseModel):
    """Send Audio Data Payload"""
    transport: str = Field(default="binary", description="Transport method: binary/base64")
    binary_size: Optional[int] = Field(default=None, description="Total binary size in binary mode")
    segment_num: Optional[int] = Field(default=None, description="Binary frame count in binary mode")
    data_base64: Optional[str] = Field(default=None, description="Base64 data")
    
    # Audio format (override session config at runtime)
    format: Optional[str] = Field(default=None, description="Audio format: PCM or OPUS")


class SendHumanAudio(BaseModel):
    """Send User Audio Data"""
    header: MessageHeader
    payload: SendHumanAudioPayload


class SendHumanVideoPayload(BaseModel):
    """Send Video Data Payload"""
    width: int = Field(..., description="Video width")
    height: int = Field(..., description="Video height")
    format: str = Field(default="JPEG", description="Video format")
    transport: str = Field(default="binary", description="Transport method: binary/base64")
    binary_size: Optional[int] = Field(default=None, description="Total binary size in binary mode")
    segment_num: Optional[int] = Field(default=None, description="Binary frame count in binary mode")
    data_base64: Optional[str] = Field(default=None, description="Base64 data")


class SendHumanVideo(BaseModel):
    """Send User Video Data"""
    header: MessageHeader
    payload: SendHumanVideoPayload


class SendHumanTextPayload(BaseModel):
    """Send Text Data Payload"""
    stream_key: str = Field(..., description="Unique stream ID")
    mode: str = Field(default="increment", description="Text mode: increment/full_text")
    text: str = Field(..., description="Text content")
    end_of_speech: bool = Field(..., description="Whether speech ended")


class SendHumanText(BaseModel):
    """Send User Text Data"""
    header: MessageHeader
    payload: SendHumanTextPayload


class TriggerHeartbeat(BaseModel):
    """Trigger Heartbeat"""
    header: MessageHeader


class Interrupt(BaseModel):
    """Interrupt Signal"""
    header: MessageHeader


# ============================================================================
# Input Port - Server -> Client Messages
# ============================================================================

class AvatarSessionInitialized(BaseModel):
    """Session Initialization Complete"""
    header: MessageHeader


class EchoTextPayload(BaseModel):
    """Text Echo Payload"""
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID: stream_{builder_id}_{stream_id}")
    mode: str = Field(default="increment", description="Text mode: increment/full_text")
    text: str = Field(..., description="Text content")
    end_of_speech: bool = Field(..., description="Whether speech ended")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom metadata added by handler"
    )


class EchoHumanText(BaseModel):
    """Echo User Text (ASR Result)"""
    header: MessageHeader
    payload: EchoTextPayload


class EchoAvatarText(BaseModel):
    """Echo Avatar Text (LLM Result)"""
    header: MessageHeader
    payload: EchoTextPayload


class EchoAvatarAudioPayload(BaseModel):
    """Echo Avatar Audio Payload"""
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID: stream_{builder_id}_{stream_id}")
    transport: str = Field(default="binary", description="Transport method: binary/base64")
    binary_size: Optional[int] = Field(default=None, description="Total binary size in binary mode")
    segment_num: Optional[int] = Field(default=None, description="Binary frame count in binary mode")
    format: str = Field(default="PCM", description="Audio format: PCM or OPUS")
    sample_rate: int = Field(default=24000, description="Audio sample rate")
    channels: int = Field(default=1, description="Audio channels")
    data_base64: Optional[str] = Field(default=None, description="Base64 data")
    end_of_speech: bool = Field(default=False, description="Whether this audio segment ended")
    
    # Opus specific fields
    opus_frame_size_ms: Optional[int] = Field(
        default=None, 
        description="Opus frame duration in ms (valid only when format=OPUS)"
    )
    
    # Extended metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom metadata added by handler"
    )


class EchoAvatarAudio(BaseModel):
    """Echo Avatar Audio (TTS Result)"""
    header: MessageHeader
    payload: EchoAvatarAudioPayload


class AvatarHeartbeat(BaseModel):
    """Heartbeat Response"""
    header: MessageHeader


class InterruptAcceptedPayload(BaseModel):
    """Interrupt Accepted Payload"""
    target_stream_id: Optional[str] = Field(default=None, description="Interrupted AVATAR_AUDIO stream ID")
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID: stream_{builder_id}_{stream_id}")


class InterruptAccepted(BaseModel):
    """Interrupt Accepted"""
    header: MessageHeader
    payload: Optional[InterruptAcceptedPayload] = Field(default=None)


class InterruptNotificationPayload(BaseModel):
    """Server Interrupt Notification Payload (for non-Motion Data mode)"""
    target_stream_id: str = Field(..., description="Interrupted AVATAR_AUDIO stream ID")
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID: stream_{builder_id}_{stream_id}")
    reason: str = Field(default="user_interrupt", description="Interrupt reason: user_interrupt | semantic_interrupt")
    interrupted_at: float = Field(default_factory=time.time, description="Interrupt timestamp")


class InterruptNotification(BaseModel):
    """Server Interrupt Notification (for non-Motion Data mode)"""
    header: MessageHeader
    payload: InterruptNotificationPayload


class ChatSignalPayload(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    type: str
    source_type: str
    stream_type: Optional[str] = Field(default=None, description="Stream type")
    stream_producer: Optional[str] = Field(default=None, description="Stream producer")
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID: stream_{builder_id}_{stream_id}")
    parent_stream_keys: Optional[List[str]] = Field(default=None, description="Parent stream_key list")
    signal_data: Optional[Dict] = Field(default=None, description="Signal data")


class ChatSignalMessage(BaseModel):
    """Observed Chat Signal"""
    header: MessageHeader
    payload: ChatSignalPayload


class ErrorCode(str, Enum):
    """Error Code Enumeration"""
    INVALID_SESSION = "INVALID_SESSION"
    AUDIO_FORMAT_ERROR = "AUDIO_FORMAT_ERROR"
    VIDEO_FORMAT_ERROR = "VIDEO_FORMAT_ERROR"
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    INVALID_MESSAGE = "INVALID_MESSAGE"
    BINARY_DATA_ERROR = "BINARY_DATA_ERROR"


class ErrorPayload(BaseModel):
    """Error Payload"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")


class Error(BaseModel):
    """Error Message"""
    header: MessageHeader
    payload: ErrorPayload


# ============================================================================
# Output Port - Server -> Renderer Messages
# ============================================================================

class MotionDataPayload(BaseModel):
    """Motion Data Payload"""
    stream_key: Optional[str] = Field(default=None, description="Unique stream ID")
    motion_data: BinaryDataInfo
    end_of_speech: bool = Field(..., description="Whether speech ended")


class MotionDataMessage(BaseModel):
    """Motion Data Message"""
    header: MessageHeader
    payload: MotionDataPayload


# ============================================================================
# Input Port - Renderer -> Server Messages
# ============================================================================

class EndSpeechPayload(BaseModel):
    """EndSpeech Payload"""
    stream_key: str = Field(..., description="Unique stream ID")


class EndSpeech(BaseModel):
    """Renderer Finished Playback"""
    header: MessageHeader
    payload: EndSpeechPayload


# ============================================================================
# Helper Functions
# ============================================================================

def parse_message(json_data: dict) -> Optional[BaseMessage]:
    """
    Parse JSON message.
    
    Args:
        json_data: JSON dictionary
        
    Returns:
        Parsed message object, or None if parsing fails.
    """
    try:
        if "header" not in json_data:
            logger.warning("JSON message missing header field")
            return None
        
        message_name = json_data["header"].get("name")
        if not message_name:
            logger.warning("JSON message header missing name field")
            return None
        
        try:
            message_type = MessageType(message_name)
        except ValueError:
            logger.warning(f"Unknown message type: {message_name}")
            return None
        
        message_class_map = {
            MessageType.INITIALIZE_AVATAR_SESSION: InitializeAvatarSession,
            MessageType.SEND_HUMAN_AUDIO: SendHumanAudio,
            MessageType.SEND_HUMAN_VIDEO: SendHumanVideo,
            MessageType.SEND_HUMAN_TEXT: SendHumanText,
            MessageType.TRIGGER_HEARTBEAT: TriggerHeartbeat,
            MessageType.INTERRUPT: Interrupt,
            MessageType.AVATAR_SESSION_INITIALIZED: AvatarSessionInitialized,
            MessageType.ECHO_HUMAN_TEXT: EchoHumanText,
            MessageType.ECHO_AVATAR_TEXT: EchoAvatarText,
            MessageType.ECHO_AVATAR_AUDIO: EchoAvatarAudio,
            MessageType.AVATAR_HEARTBEAT: AvatarHeartbeat,
            MessageType.INTERRUPT_ACCEPTED: InterruptAccepted,
            MessageType.ERROR: Error,
            MessageType.MOTION_DATA: MotionDataMessage,
            MessageType.MOTION_DATA_WELCOME: MotionDataMessage,
            MessageType.END_SPEECH: EndSpeech,
        }
        
        message_class = message_class_map.get(message_type)
        if message_class is None:
            logger.error(f"Message type {message_type} has no handler class")
            return None
        
        return message_class.model_validate(json_data)
    except Exception as e:
        logger.error(f"Exception while parsing JSON message: {e}", exc_info=True)
        return None


def serialize_message(message: BaseMessage) -> dict:
    """
    Serialize message to JSON dictionary.
    
    Args:
        message: Message object
        
    Returns:
        JSON dictionary
    """
    return message.model_dump(exclude_none=True)

