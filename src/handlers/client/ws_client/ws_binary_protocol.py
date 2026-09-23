"""
WebSocket Binary Protocol Handling
Handles packing and unpacking of binary data.
"""
import struct
from typing import Optional, List, Tuple
from dataclasses import dataclass
from loguru import logger


# ============================================================================
# Binary Packet Header Structure
# ============================================================================

@dataclass
class BinaryPacketHeader:
    """
    Binary Packet Header (36 bytes)
    
    Header (36 bytes):
      - Magic: "JBIN" (4 bytes)
      - Request Id: "xxx" (8 bytes, first 8 characters of request_id)
      - Packet Type: 0 (4 bytes, 0=audio, 1=video)
      - Packet Index: 0 (4 bytes, current packet index)
      - Total Packets: 2 (4 bytes, total packet count)
      - Data Size: 20000 (4 bytes, current packet data size)
      - Reserved: 0 (8 bytes, reserved field)
    """
    magic: bytes  # 4 bytes: b"JBIN"
    request_id: str  # 8 bytes: first 8 chars of request_id
    packet_type: int  # 4 bytes: 0=audio, 1=video
    packet_index: int  # 4 bytes: current packet index
    total_packets: int  # 4 bytes: total packet count
    data_size: int  # 4 bytes: current packet data size
    reserved: int  # 8 bytes: reserved field
    
    HEADER_SIZE = 36
    MAGIC = b"JBIN"
    
    @classmethod
    def parse(cls, data: bytes) -> Optional['BinaryPacketHeader']:
        """
        Parse binary packet header.
        
        Args:
            data: At least 36 bytes of data.
            
        Returns:
            Parsed header object, or None if parsing fails.
        """
        if len(data) < cls.HEADER_SIZE:
            return None
        
        try:
            # Unpack: 4s(magic) + 8s(request_id) + 4I(4x uint32) + Q(uint64)
            unpacked = struct.unpack("<4s8s4IQ", data[:cls.HEADER_SIZE])
            
            magic = unpacked[0]
            if magic != cls.MAGIC:
                logger.warning(f"Invalid magic: {magic}, expected {cls.MAGIC}")
                return None
            
            request_id = unpacked[1].decode('utf-8', errors='ignore').rstrip('\x00')
            packet_type = unpacked[2]
            packet_index = unpacked[3]
            total_packets = unpacked[4]
            data_size = unpacked[5]
            reserved = unpacked[6]
            
            return cls(
                magic=magic,
                request_id=request_id,
                packet_type=packet_type,
                packet_index=packet_index,
                total_packets=total_packets,
                data_size=data_size,
                reserved=reserved
            )
        except Exception as e:
            logger.error(f"Failed to parse binary packet header: {e}")
            return None
    
    def pack(self) -> bytes:
        """
        Pack header into binary data.
        
        Returns:
            36-byte binary data.
        """
        # Ensure request_id is 8 bytes
        request_id_bytes = self.request_id[:8].encode('utf-8')
        request_id_bytes = request_id_bytes.ljust(8, b'\x00')
        
        return struct.pack(
            "<4s8s4IQ",
            self.magic,
            request_id_bytes,
            self.packet_type,
            self.packet_index,
            self.total_packets,
            self.data_size,
            self.reserved
        )


# ============================================================================
# Binary Packet Assembler (for receiving segmented data)
# ============================================================================

@dataclass
class StreamAssemblyState:
    """Human audio/video binary stream assembly state."""
    request_id: str
    expected_segments: int
    expected_size: int
    metadata: Optional[object] = None
    received_segments: int = 0
    received_size: int = 0
    chunks: List[bytes] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []
    
    def append(self, chunk: bytes) -> Optional[bytes]:
        """Append a segment; return complete binary data when finished."""
        self.chunks.append(chunk)
        self.received_segments += 1
        self.received_size += len(chunk)
        
        if self.received_segments >= self.expected_segments or self.received_size >= self.expected_size:
            data = b"".join(self.chunks)
            if self.expected_size and self.expected_size != len(data):
                logger.warning(
                    f"Binary data size mismatch for request_id={self.request_id}: "
                    f"expected={self.expected_size}, actual={len(data)}"
                )
            return data
        return None


class BinaryStreamAssembler:
    """
    Binary Stream Assembler
    Assembles header-less binary segments in registration order (used for base audio/video upload).
    """
    
    def __init__(self):
        from collections import deque
        self._queue = deque()  # type: ignore[var-annotated]
    
    def register(self, request_id: str, expected_segments: int, expected_size: int,
                 metadata: Optional[object] = None) -> StreamAssemblyState:
        state = StreamAssemblyState(
            request_id=request_id,
            expected_segments=expected_segments,
            expected_size=expected_size,
            metadata=metadata or {}
        )
        self._queue.append(state)
        return state
    
    def append(self, chunk: bytes) -> Optional[Tuple[StreamAssemblyState, bytes]]:
        if not self._queue:
            logger.warning("Received unexpected binary chunk with no pending registrations")
            return None
        
        state = self._queue[0]
        data = state.append(chunk)
        if data is not None:
            self._queue.popleft()
            return state, data
        return None
    
    def clear(self):
        self._queue.clear()


# ============================================================================
# Binary Packet Splitter (for sending large data)
# ============================================================================

class BinaryPacketSplitter:
    """
    Binary Packet Splitter
    Splits large data into fixed-size segments.
    """
    
    # Motion Data output segment size: 16KB
    MOTION_DATA_SEGMENT_SIZE = 16 * 1024
    
    @staticmethod
    def split(request_id: str, packet_type: int, data: bytes, 
              segment_size: int = MOTION_DATA_SEGMENT_SIZE) -> List[bytes]:
        """
        Split data into multiple segments.
        
        Args:
            request_id: Request ID.
            packet_type: Data type (0=audio, 1=video).
            data: Data to split.
            segment_size: Size of each segment (excluding header).
            
        Returns:
            List of segments, each containing header and data.
        """
        if len(data) == 0:
            # Empty data, return an empty packet
            return [b""]
        
        # Calculate total segment count
        total_packets = (len(data) + segment_size - 1) // segment_size
        
        packets = []
        for i in range(total_packets):
            start = i * segment_size
            end = min(start + segment_size, len(data))
            segment_data = data[start:end]
            
            packet = segment_data
            packets.append(packet)
        
        return packets

