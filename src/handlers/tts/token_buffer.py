"""
Token Buffer for TTS - Intelligent handling of LLM token-by-token output concatenation

Design Principles:
1. Space-separated languages (English, French, German, etc.): Wait for space/punctuation/CJK character before emitting, preventing word truncation.
2. Non-space-separated languages (Chinese, Japanese, Korean, Thai, etc.): Each character can be pronounced independently and sent immediately.
3. Mixed languages: Intelligently switch processing strategies (e.g. mixed Chinese-English like "Today is a sunny day").

Example Usage:
    buffer = TokenBuffer()
    for token in llm_tokens:
        text_to_send = buffer.process(token)
        if text_to_send:
            tts.send(text_to_send)
    # Flush remaining content on stream end
    remaining = buffer.flush()
    if remaining:
        tts.send(remaining)
"""

import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


class CharType(Enum):
    """Character type classification"""
    CJK = auto()           # CJK Unified Ideographs (immediate send)
    HANGUL = auto()        # Korean Hangul syllables (immediate send)
    THAI = auto()          # Thai script (immediate send)
    KANA = auto()          # Japanese Kana (immediate send)
    LATIN = auto()         # Latin letters (space-delimited)
    CYRILLIC = auto()      # Cyrillic letters (space-delimited)
    ARABIC = auto()        # Arabic script (space-delimited)
    HEBREW = auto()        # Hebrew script (space-delimited)
    GREEK = auto()         # Greek script (space-delimited)
    SPACE = auto()         # Whitespace character
    PUNCTUATION = auto()   # Punctuation mark
    NUMBER = auto()        # Numeric digit
    OTHER = auto()         # Other characters


def get_char_type(char: str) -> CharType:
    """
    Determine character type based on Unicode range.
    
    Args:
        char: Single character
        
    Returns:
        CharType enum value
    """
    if not char:
        return CharType.OTHER
    
    code = ord(char)
    
    # Whitespace
    if char.isspace():
        return CharType.SPACE
    
    # CJK Unified Ideographs (Chinese, Japanese Kanji, Korean Hanja)
    if (0x4E00 <= code <= 0x9FFF or      # CJK Unified Ideographs
        0x3400 <= code <= 0x4DBF or      # CJK Extension A
        0x20000 <= code <= 0x2A6DF or    # CJK Extension B
        0x2A700 <= code <= 0x2B73F or    # CJK Extension C
        0x2B740 <= code <= 0x2B81F or    # CJK Extension D
        0x2B820 <= code <= 0x2CEAF or    # CJK Extension E
        0x2CEB0 <= code <= 0x2EBEF or    # CJK Extension F
        0x30000 <= code <= 0x3134F or    # CJK Extension G
        0xF900 <= code <= 0xFAFF or      # CJK Compatibility Ideographs
        0x2F800 <= code <= 0x2FA1F):     # CJK Compatibility Supplement
        return CharType.CJK
    
    # Japanese Kana
    if (0x3040 <= code <= 0x309F or      # Hiragana
        0x30A0 <= code <= 0x30FF or      # Katakana
        0x31F0 <= code <= 0x31FF or      # Katakana Phonetic Extensions
        0xFF65 <= code <= 0xFF9F):       # Halfwidth Katakana
        return CharType.KANA
    
    # Korean Hangul
    if (0xAC00 <= code <= 0xD7AF or      # Hangul Syllables
        0x1100 <= code <= 0x11FF or      # Hangul Jamo
        0x3130 <= code <= 0x318F or      # Hangul Compatibility Jamo
        0xA960 <= code <= 0xA97F or      # Hangul Jamo Extended-A
        0xD7B0 <= code <= 0xD7FF):       # Hangul Jamo Extended-B
        return CharType.HANGUL
    
    # Thai
    if 0x0E00 <= code <= 0x0E7F:         # Thai
        return CharType.THAI
    
    # Latin letters
    if (0x0041 <= code <= 0x007A or      # Basic Latin letters
        0x00C0 <= code <= 0x00FF or      # Latin-1 Supplement
        0x0100 <= code <= 0x017F or      # Latin Extended-A
        0x0180 <= code <= 0x024F or      # Latin Extended-B
        0x1E00 <= code <= 0x1EFF):       # Latin Extended Additional
        return CharType.LATIN
    
    # Cyrillic letters (Russian, etc.)
    if (0x0400 <= code <= 0x04FF or      # Cyrillic
        0x0500 <= code <= 0x052F):       # Cyrillic Supplement
        return CharType.CYRILLIC
    
    # Arabic script
    if (0x0600 <= code <= 0x06FF or      # Arabic
        0x0750 <= code <= 0x077F or      # Arabic Supplement
        0x08A0 <= code <= 0x08FF):       # Arabic Extended-A
        return CharType.ARABIC
    
    # Hebrew script
    if 0x0590 <= code <= 0x05FF:         # Hebrew
        return CharType.HEBREW
    
    # Greek script
    if (0x0370 <= code <= 0x03FF or      # Greek and Coptic
        0x1F00 <= code <= 0x1FFF):       # Greek Extended
        return CharType.GREEK
    
    # Numbers
    if char.isdigit():
        return CharType.NUMBER
    
    # Punctuation (using Unicode category)
    category = unicodedata.category(char)
    if category.startswith('P') or category.startswith('S'):
        return CharType.PUNCTUATION
    
    return CharType.OTHER


def is_immediate_sendable(char_type: CharType) -> bool:
    """
    Check whether characters of this type can be sent immediately without waiting.
    """
    return char_type in {
        CharType.CJK,
        CharType.KANA,
        CharType.HANGUL,
        CharType.THAI,
        CharType.SPACE,
        CharType.PUNCTUATION,
    }


def is_word_char(char_type: CharType) -> bool:
    """
    Check whether characters of this type require word-boundary (space) separation.
    """
    return char_type in {
        CharType.LATIN,
        CharType.CYRILLIC,
        CharType.ARABIC,
        CharType.HEBREW,
        CharType.GREEK,
        CharType.NUMBER,
    }


@dataclass
class TokenBuffer:
    """
    Token Buffer - Intelligent handling of LLM token streams
    """
    buffer: str = ""
    min_buffer_chars: int = 0
    max_buffer_chars: int = 200
    
    def process(self, token: Optional[str]) -> str:
        """
        Process input token and return text ready to send to TTS.
        
        Args:
            token: Input token string (may be None or empty)
            
        Returns:
            Text ready for TTS (may be empty string if held in buffer)
        """
        if not token:
            return ""
        
        result = []
        
        for char in token:
            char_type = get_char_type(char)
            
            if is_immediate_sendable(char_type):
                if self.buffer:
                    result.append(self.buffer)
                    self.buffer = ""
                result.append(char)
            elif is_word_char(char_type):
                self.buffer += char
                if len(self.buffer) >= self.max_buffer_chars:
                    result.append(self.buffer)
                    self.buffer = ""
            else:
                self.buffer += char
        
        output = "".join(result)
        
        if output and len(output) < self.min_buffer_chars:
            self.buffer = output + self.buffer
            return ""
        
        return output
    
    def flush(self) -> str:
        """Flush buffer and return all remaining text."""
        result = self.buffer
        self.buffer = ""
        return result
    
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer = ""
    
    def peek(self) -> str:
        """Peek current buffer content without clearing."""
        return self.buffer
    
    def __len__(self) -> int:
        """Return current buffer length."""
        return len(self.buffer)


@dataclass
class SentenceAwareTokenBuffer(TokenBuffer):
    """
    Sentence-aware Token Buffer
    
    Extends TokenBuffer to split text at sentence/clause boundaries.
    """
    sentence_delimiters: set = field(default_factory=lambda: {
        '。', '！', '？', '；',  # Chinese punctuation
        '.', '!', '?', ';',    # English punctuation
        '।', '؟', '।',         # Other punctuation
    })
    clause_delimiters: set = field(default_factory=lambda: {
        '，', '：', '、',       # Chinese punctuation
        ',', ':',              # English punctuation
    })
    split_on_clause: bool = False
    _pending_output: str = field(default="", init=False)
    
    def process(self, token: Optional[str]) -> str:
        """Process input token and return complete sentences at sentence boundaries."""
        if not token:
            return ""
        
        immediate_output = super().process(token)
        self._pending_output += immediate_output
        
        result = []
        remaining = ""
        
        i = 0
        last_split = 0
        
        while i < len(self._pending_output):
            char = self._pending_output[i]
            
            is_sentence_end = char in self.sentence_delimiters
            is_clause_end = self.split_on_clause and char in self.clause_delimiters
            
            if is_sentence_end or is_clause_end:
                result.append(self._pending_output[last_split:i + 1])
                last_split = i + 1
            
            i += 1
        
        remaining = self._pending_output[last_split:]
        self._pending_output = remaining
        
        return "".join(result)
    
    def flush(self) -> str:
        """Flush all buffered content."""
        parent_flush = super().flush()
        result = self._pending_output + parent_flush
        self._pending_output = ""
        return result
    
    def clear(self) -> None:
        """Clear all buffers."""
        super().clear()
        self._pending_output = ""


def create_token_buffer(
    sentence_aware: bool = False,
    split_on_clause: bool = False,
    min_buffer_chars: int = 0,
    max_buffer_chars: int = 200,
) -> TokenBuffer:
    """Factory function to create a TokenBuffer instance."""
    if sentence_aware:
        return SentenceAwareTokenBuffer(
            min_buffer_chars=min_buffer_chars,
            max_buffer_chars=max_buffer_chars,
            split_on_clause=split_on_clause,
        )
    else:
        return TokenBuffer(
            min_buffer_chars=min_buffer_chars,
            max_buffer_chars=max_buffer_chars,
        )
