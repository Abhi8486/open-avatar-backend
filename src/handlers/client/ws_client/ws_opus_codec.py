"""
WebSocket Opus Codec
Provides Opus audio encoding and decoding functionality for WebSocket audio transmission.
"""
import struct
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np
from loguru import logger

# Try importing opuslib
# Note: opuslib throws a generic Exception (not ImportError) on Windows if opus.dll is missing.
# Catch Exception here to avoid breaking startup for RTC-only flows.
try:
    import opuslib
    OPUS_AVAILABLE = True
except Exception as _opus_import_error:
    opuslib = None  # type: ignore[assignment]
    OPUS_AVAILABLE = False
    logger.warning(
        f"opuslib not available, Opus codec will be disabled: {_opus_import_error}"
    )


# ============================================================================
# Opus Configuration Constants
# ============================================================================

# Opus supported frame sizes (in sample points for given sample rates)
# For 48kHz: 2.5ms=120, 5ms=240, 10ms=480, 20ms=960, 40ms=1920, 60ms=2880
# For 24kHz: 2.5ms=60, 5ms=120, 10ms=240, 20ms=480, 40ms=960, 60ms=1440
# For 16kHz: 2.5ms=40, 5ms=80, 10ms=160, 20ms=320, 40ms=640, 60ms=960
OPUS_FRAME_DURATIONS_MS = [2.5, 5, 10, 20, 40, 60]

# Default configuration
DEFAULT_OPUS_SAMPLE_RATE = 48000  # Opus internal sample rate (resampled automatically during encoding)
DEFAULT_OPUS_CHANNELS = 1
DEFAULT_OPUS_APPLICATION = 'voip'  # 'voip', 'audio', or 'restricted_lowdelay'
DEFAULT_OPUS_BITRATE = 32000  # 32 kbps, sufficient for speech
DEFAULT_OPUS_FRAME_SIZE_MS = 20  # 20ms per frame


@dataclass
class OpusFrameHeader:
    """
    Opus Frame Header (4 bytes)
    
    Used to encapsulate a single Opus frame in a binary stream:
      - frame_size: uint16 (2 bytes, Little Endian) - Frame data size (excluding header)
      - frame_duration_samples: uint16 (2 bytes, Little Endian) - Frame duration (sample points)
    
    Example: For 48kHz, 20ms frame, frame_duration_samples = 960
    """
    frame_size: int  # Frame data size (bytes)
    frame_duration_samples: int  # Frame duration (sample points)
    
    HEADER_SIZE = 4
    
    def pack(self) -> bytes:
        """Pack header into binary data."""
        return struct.pack("<HH", self.frame_size, self.frame_duration_samples)
    
    @classmethod
    def unpack(cls, data: bytes) -> Optional['OpusFrameHeader']:
        """Unpack header from binary data."""
        if len(data) < cls.HEADER_SIZE:
            return None
        try:
            frame_size, frame_duration_samples = struct.unpack("<HH", data[:cls.HEADER_SIZE])
            return cls(frame_size=frame_size, frame_duration_samples=frame_duration_samples)
        except Exception as e:
            logger.error(f"Failed to unpack Opus frame header: {e}")
            return None


@dataclass
class OpusStreamHeader:
    """
    Opus Stream Header (8 bytes)
    
    Describes parameters for the entire Opus data stream:
      - magic: "OPUS" (4 bytes, ASCII)
      - sample_rate: uint16 (2 bytes, Little Endian) - Original audio sample rate / 100
      - channels: uint8 (1 byte) - Number of channels
      - frame_size_ms: uint8 (1 byte) - Frame duration (milliseconds)
    
    Note: sample_rate is stored in units of 100, e.g. 16000Hz is stored as 160.
    """
    sample_rate: int  # Original audio sample rate (Hz)
    channels: int  # Number of channels
    frame_size_ms: int  # Frame duration (ms)
    
    HEADER_SIZE = 8
    MAGIC = b"OPUS"
    
    def pack(self) -> bytes:
        """Pack header into binary data."""
        sample_rate_encoded = self.sample_rate // 100
        return struct.pack(
            "<4sHBB",
            self.MAGIC,
            sample_rate_encoded,
            self.channels,
            self.frame_size_ms
        )
    
    @classmethod
    def unpack(cls, data: bytes) -> Optional['OpusStreamHeader']:
        """Unpack header from binary data."""
        if len(data) < cls.HEADER_SIZE:
            return None
        try:
            magic, sample_rate_encoded, channels, frame_size_ms = struct.unpack(
                "<4sHBB", data[:cls.HEADER_SIZE]
            )
            if magic != cls.MAGIC:
                logger.warning(f"Invalid Opus stream magic: {magic}, expected {cls.MAGIC}")
                return None
            return cls(
                sample_rate=sample_rate_encoded * 100,
                channels=channels,
                frame_size_ms=frame_size_ms
            )
        except Exception as e:
            logger.error(f"Failed to unpack Opus stream header: {e}")
            return None


class OpusEncoder:
    """
    Opus Encoder
    
    Encodes PCM audio data to Opus format.
    
    Note: To prevent periodic silence artifacts, the encoder buffers sub-frame residual data
    until the next encoding call. Residual data is only flushed (zero-padded to a full frame)
    when calling flush() or encode(..., flush=True).
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        channels: int = 1,
        application: str = DEFAULT_OPUS_APPLICATION,
        bitrate: int = DEFAULT_OPUS_BITRATE,
        frame_size_ms: int = DEFAULT_OPUS_FRAME_SIZE_MS
    ):
        """
        Initialize Opus Encoder.
        
        Args:
            sample_rate: Input audio sample rate (Hz), supports 8000, 12000, 16000, 24000, 48000
            channels: Number of channels (1 or 2)
            application: Application type ('voip', 'audio', 'restricted_lowdelay')
            bitrate: Target bitrate (bps)
            frame_size_ms: Frame duration (ms), supports 2.5, 5, 10, 20, 40, 60
        """
        if not OPUS_AVAILABLE:
            raise RuntimeError("opuslib is not available. Please install it with: pip install opuslib")
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.application = application
        self.bitrate = bitrate
        self.frame_size_ms = frame_size_ms
        
        # Calculate frame size (sample count)
        self.frame_size_samples = int(sample_rate * frame_size_ms / 1000)
        
        # Create encoder
        app_map = {
            'voip': opuslib.APPLICATION_VOIP,
            'audio': opuslib.APPLICATION_AUDIO,
            'restricted_lowdelay': opuslib.APPLICATION_RESTRICTED_LOWDELAY
        }
        self._encoder = opuslib.Encoder(sample_rate, channels, app_map.get(application, opuslib.APPLICATION_VOIP))
        
        # Residual buffer - buffers data less than one full frame
        self._residual_buffer: Optional[np.ndarray] = None
        
        logger.info(
            f"Opus encoder initialized: sample_rate={sample_rate}, channels={channels}, "
            f"frame_size_ms={frame_size_ms}, frame_size_samples={self.frame_size_samples}"
        )
    
    def encode(self, pcm_data: np.ndarray, flush: bool = False) -> bytes:
        """
        Encode PCM data into an Opus stream.
        
        Args:
            pcm_data: PCM audio data, shape [N] or [channels, N], dtype int16 or float32.
            flush: Force flush residual buffer (set True at stream end).
            
        Returns:
            Encoded Opus data containing stream header and frame data.
            
        Note:
            - Sub-frame residual data is buffered for the next call.
            - Residual data is zero-padded only when flush=True.
            - If input data + residual is still sub-frame and flush=False, returns empty Opus stream.
        """
        # Ensure 1D array
        if pcm_data.ndim > 1:
            pcm_data = pcm_data.flatten()
        
        # Convert to int16
        if pcm_data.dtype != np.int16:
            if np.issubdtype(pcm_data.dtype, np.floating):
                pcm_data = np.clip(pcm_data, -1.0, 1.0)
                pcm_data = (pcm_data * 32767).astype(np.int16)
            else:
                pcm_data = pcm_data.astype(np.int16)
        
        # Concatenate residual buffer
        if self._residual_buffer is not None and len(self._residual_buffer) > 0:
            pcm_data = np.concatenate([self._residual_buffer, pcm_data])
            self._residual_buffer = None
        
        # Build stream header
        stream_header = OpusStreamHeader(
            sample_rate=self.sample_rate,
            channels=self.channels,
            frame_size_ms=self.frame_size_ms
        )
        
        # Encode frames
        encoded_frames = []
        total_samples = len(pcm_data)
        offset = 0
        
        while offset < total_samples:
            remaining = total_samples - offset
            
            if remaining < self.frame_size_samples:
                # Sub-frame data
                if flush:
                    # Force flush: zero-pad to full frame
                    frame_pcm = np.pad(pcm_data[offset:], (0, self.frame_size_samples - remaining))
                    encoded_data = self._encoder.encode(frame_pcm.tobytes(), self.frame_size_samples)
                    frame_header = OpusFrameHeader(
                        frame_size=len(encoded_data),
                        frame_duration_samples=self.frame_size_samples
                    )
                    encoded_frames.append(frame_header.pack() + encoded_data)
                    logger.debug(f"Flushed residual {remaining} samples with zero-padding")
                else:
                    # Buffer residual data
                    self._residual_buffer = pcm_data[offset:].copy()
                    logger.debug(f"Buffered {remaining} residual samples for next encode")
                break
            
            # Full frame: encode directly
            frame_pcm = pcm_data[offset:offset + self.frame_size_samples]
            encoded_data = self._encoder.encode(frame_pcm.tobytes(), self.frame_size_samples)
            
            frame_header = OpusFrameHeader(
                frame_size=len(encoded_data),
                frame_duration_samples=self.frame_size_samples
            )
            
            encoded_frames.append(frame_header.pack() + encoded_data)
            offset += self.frame_size_samples
        
        # Combine stream header and frames
        result = stream_header.pack() + b"".join(encoded_frames)
        
        logger.debug(
            f"Opus encoded: {total_samples} samples -> {len(result)} bytes, "
            f"{len(encoded_frames)} frames, residual={len(self._residual_buffer) if self._residual_buffer is not None else 0}"
        )
        
        return result
    
    def flush(self) -> bytes:
        """
        Flush residual buffer and encode remaining data.
        
        Returns:
            Encoded Opus data (if residual existed), else empty stream.
        """
        if self._residual_buffer is None or len(self._residual_buffer) == 0:
            # Return empty Opus stream
            stream_header = OpusStreamHeader(
                sample_rate=self.sample_rate,
                channels=self.channels,
                frame_size_ms=self.frame_size_ms
            )
            return stream_header.pack()
        
        # Call encode with empty array and flush=True
        return self.encode(np.array([], dtype=np.int16), flush=True)
    
    def reset(self):
        """
        Reset encoder state and clear residual buffer.
        
        Call when starting a new audio stream.
        """
        self._residual_buffer = None
        logger.debug("Opus encoder reset")
    
    def encode_frame(self, pcm_frame: np.ndarray) -> bytes:
        """
        Encode a single PCM frame (excluding stream header).
        
        Args:
            pcm_frame: Single frame PCM data, length should be frame_size_samples.
            
        Returns:
            Encoded frame data (including frame header).
        """
        # Convert to int16
        if pcm_frame.dtype != np.int16:
            if np.issubdtype(pcm_frame.dtype, np.floating):
                pcm_frame = np.clip(pcm_frame, -1.0, 1.0)
                pcm_frame = (pcm_frame * 32767).astype(np.int16)
            else:
                pcm_frame = pcm_frame.astype(np.int16)
        
        # Ensure correct frame length
        if len(pcm_frame) < self.frame_size_samples:
            pcm_frame = np.pad(pcm_frame, (0, self.frame_size_samples - len(pcm_frame)))
        elif len(pcm_frame) > self.frame_size_samples:
            pcm_frame = pcm_frame[:self.frame_size_samples]
        
        # Encode
        encoded_data = self._encoder.encode(pcm_frame.tobytes(), self.frame_size_samples)
        
        # Build frame header
        frame_header = OpusFrameHeader(
            frame_size=len(encoded_data),
            frame_duration_samples=self.frame_size_samples
        )
        
        return frame_header.pack() + encoded_data


class OpusDecoder:
    """
    Opus Decoder
    
    Decodes Opus format data to PCM audio.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1
    ):
        """
        Initialize Opus Decoder.
        
        Args:
            sample_rate: Output audio sample rate (Hz).
            channels: Number of channels (1 or 2).
        """
        if not OPUS_AVAILABLE:
            raise RuntimeError("opuslib is not available. Please install it with: pip install opuslib")
        
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Create decoder
        self._decoder = opuslib.Decoder(sample_rate, channels)
        
        logger.info(f"Opus decoder initialized: sample_rate={sample_rate}, channels={channels}")
    
    def decode(self, opus_data: bytes) -> np.ndarray:
        """
        Decode an Opus stream to PCM data.
        
        Args:
            opus_data: Complete Opus data stream (containing stream header and frame data).
            
        Returns:
            Decoded PCM data, dtype int16.
        """
        if len(opus_data) < OpusStreamHeader.HEADER_SIZE:
            raise ValueError(f"Opus data too short: {len(opus_data)} bytes")
        
        # Parse stream header
        stream_header = OpusStreamHeader.unpack(opus_data)
        if stream_header is None:
            raise ValueError("Failed to parse Opus stream header")
        
        # Parse frame data
        offset = OpusStreamHeader.HEADER_SIZE
        decoded_frames = []
        
        while offset < len(opus_data):
            # Parse frame header
            frame_header = OpusFrameHeader.unpack(opus_data[offset:])
            if frame_header is None:
                logger.warning(f"Failed to parse Opus frame header at offset {offset}")
                break
            
            offset += OpusFrameHeader.HEADER_SIZE
            
            # Read frame data
            frame_data = opus_data[offset:offset + frame_header.frame_size]
            if len(frame_data) < frame_header.frame_size:
                logger.warning(f"Incomplete Opus frame: expected {frame_header.frame_size}, got {len(frame_data)}")
                break
            
            offset += frame_header.frame_size
            
            # Calculate target sample points
            target_samples = int(
                frame_header.frame_duration_samples * self.sample_rate / stream_header.sample_rate
            )
            
            # Decode
            pcm_bytes = self._decoder.decode(frame_data, target_samples)
            pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            decoded_frames.append(pcm_array)
        
        if not decoded_frames:
            return np.array([], dtype=np.int16)
        
        result = np.concatenate(decoded_frames)
        
        logger.debug(
            f"Opus decoded: {len(opus_data)} bytes -> {len(result)} samples, "
            f"{len(decoded_frames)} frames"
        )
        
        return result
    
    def decode_frame(self, frame_data: bytes, frame_duration_samples: int) -> np.ndarray:
        """
        Decode a single Opus frame (excluding frame header).
        
        Args:
            frame_data: Encoded frame data.
            frame_duration_samples: Frame duration (sample points).
            
        Returns:
            Decoded PCM data, dtype int16.
        """
        pcm_bytes = self._decoder.decode(frame_data, frame_duration_samples)
        return np.frombuffer(pcm_bytes, dtype=np.int16)


# ============================================================================
# Utility Functions
# ============================================================================

def encode_pcm_to_opus(
    pcm_data: np.ndarray,
    sample_rate: int = 24000,
    channels: int = 1,
    bitrate: int = DEFAULT_OPUS_BITRATE,
    frame_size_ms: int = DEFAULT_OPUS_FRAME_SIZE_MS
) -> bytes:
    """
    Utility function: Encode PCM data to Opus.
    
    Args:
        pcm_data: PCM audio data.
        sample_rate: Sample rate.
        channels: Number of channels.
        bitrate: Target bitrate.
        frame_size_ms: Frame duration.
        
    Returns:
        Encoded Opus data.
    """
    encoder = OpusEncoder(
        sample_rate=sample_rate,
        channels=channels,
        bitrate=bitrate,
        frame_size_ms=frame_size_ms
    )
    return encoder.encode(pcm_data)


def decode_opus_to_pcm(
    opus_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1
) -> np.ndarray:
    """
    Utility function: Decode Opus data to PCM.
    
    Args:
        opus_data: Opus audio data.
        sample_rate: Output sample rate.
        channels: Number of channels.
        
    Returns:
        Decoded PCM data.
    """
    decoder = OpusDecoder(sample_rate=sample_rate, channels=channels)
    return decoder.decode(opus_data)


def is_opus_available() -> bool:
    """Check if Opus codec is available."""
    return OPUS_AVAILABLE
