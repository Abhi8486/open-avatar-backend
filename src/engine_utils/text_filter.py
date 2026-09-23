"""
Text Filter for TTS - Utility to clean markdown text output from LLM for TTS synthesis

Design principles:
1. Remove brackets and their contents (parentheses, square brackets, curly braces, etc.)
2. Convert emphasis markers to language-appropriate quotation marks
3. Remove markdown syntax (headers, lists, links, code blocks, etc.)
4. Preserve standard conversational punctuation (commas, periods, question marks, exclamation marks, etc.)
5. Support incremental text stream processing (handles incomplete markdown markers)

Usage example:
    filter = TextFilter(
        remove_brackets=True,
        convert_emphasis_to_quotes=True,
        remove_markdown=True
    )
    filtered_text = filter.filter("This is **important** content")
    # Output: This is "important" content
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextFilterConfig:
    """
    Text filter configuration.
    
    Attributes:
        remove_brackets: Whether to remove brackets and their inner contents.
        remove_code_blocks: Whether to remove code blocks.
        remove_markdown_links: Whether to remove markdown links and images.
        remove_markdown_formatting: Whether to remove markdown formatting (headers, lists, etc.).
        convert_emphasis_to_quotes: Whether to convert emphasis markers into quotation marks.
        remove_special_tokens: Whether to remove special tokens (e.g., <|...|>).
        chinese_quote_style: Chinese quote style, e.g. '「」' or '""'.
        english_quote_style: English quote style, e.g. '"' or '''.
        chinese_threshold: Threshold ratio of Chinese characters to classify text as Chinese.
    """
    remove_brackets: bool = True
    remove_code_blocks: bool = True
    remove_markdown_links: bool = True
    remove_markdown_formatting: bool = True
    convert_emphasis_to_quotes: bool = True
    remove_special_tokens: bool = True
    chinese_quote_style: str = '「」'
    english_quote_style: str = '"'
    chinese_threshold: float = 0.3


class TextFilter:
    """
    Text filter - cleans text to retain only spoken content suitable for TTS.
    
    Supports incremental text stream processing, handling incomplete markdown tokens.
    """
    
    def __init__(self, config: Optional[TextFilterConfig] = None):
        """
        Initialize text filter.
        
        Args:
            config: Filter configuration, uses default if None.
        """
        self.config = config or TextFilterConfig()
    
    def filter(self, text: str) -> str:
        """
        Filter text, keeping only speakable content for TTS.
        
        Args:
            text: Input text (may contain markdown markers).
            
        Returns:
            Filtered text suitable for TTS readout.
        """
        if not text:
            return text
        
        # 1. Remove code blocks
        if self.config.remove_code_blocks:
            text = self._remove_code_blocks(text)
        
        # 2. Remove brackets and their inner contents
        if self.config.remove_brackets:
            text = self._remove_brackets(text)
        
        # 3. Remove markdown links and images
        if self.config.remove_markdown_links:
            text = self._remove_markdown_links(text)
        
        # 4. Remove markdown formatting markers
        if self.config.remove_markdown_formatting:
            text = self._remove_markdown_formatting(text)
        
        # 5. Remove special tokens
        if self.config.remove_special_tokens:
            text = self._remove_special_tokens(text)
        
        # 6. Process emphasis markers, converting to quote marks
        if self.config.convert_emphasis_to_quotes:
            text = self._convert_emphasis_to_quotes(text)
        
        # 7. Clean whitespace
        text = self._clean_whitespace(text)
        
        return text
    
    def _remove_code_blocks(self, text: str) -> str:
        """Remove code blocks."""
        text = re.sub(r'```[\s\S]*?```', '', text)  # Multiline code block
        text = re.sub(r'`[^`\n]*`', '', text)  # Inline code (avoid multiline matching)
        return text
    
    def _remove_brackets(self, text: str) -> str:
        """Remove brackets and their contents."""
        # Parentheses: ()、（）
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'（[^）]*）', '', text)
        # Square brackets: []、【】
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = re.sub(r'【[^】]*】', '', text)
        # Curly braces: {}、｛｝
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'｛[^｝]*｝', '', text)
        # Angle brackets: <>、〈〉、«»
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'〈[^〉]*〉', '', text)
        text = re.sub(r'«[^»]*»', '', text)
        # Other Chinese brackets: 『』、「」
        text = re.sub(r'『[^』]*』', '', text)
        text = re.sub(r'「[^」]*」', '', text)
        return text
    
    def _remove_markdown_links(self, text: str) -> str:
        """Remove markdown links and images."""
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)  # Image ![alt](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Link [text](url), preserve text
        return text
    
    def _remove_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting markers."""
        # Remove header markers (starting with #)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove list markers (starting with -, *, +)
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)  # Ordered list
        
        # Remove quote markers (starting with >)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        
        # Remove table markers
        text = re.sub(r'\|', '', text)  # Table separator
        text = re.sub(r'^[\s]*:?-+:?[\s]*$', '', text, flags=re.MULTILINE)  # Table separator row
        
        # Remove horizontal rules and header underline
        text = re.sub(r'---+', '', text)
        text = re.sub(r'===+', '', text)
        
        return text
    
    def _remove_special_tokens(self, text: str) -> str:
        """Remove special tokens."""
        text = re.sub(r'<\|.*?\|>', '', text)
        return text
    
    def _convert_emphasis_to_quotes(self, text: str) -> str:
        """Convert emphasis markers to quotation marks."""
        # Detect primary language of text (simple heuristic: ratio of Chinese characters)
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.sub(r'\s', '', text))
        is_chinese_dominant = total_chars > 0 and chinese_chars / total_chars > self.config.chinese_threshold
        
        # Select appropriate quotation marks based on language
        if is_chinese_dominant:
            if self.config.chinese_quote_style == '「」':
                quote_open = '「'
                quote_close = '」'
            else:
                quote_open = '"'
                quote_close = '"'
        else:
            if self.config.english_quote_style == '"':
                quote_open = '"'
                quote_close = '"'
            else:
                quote_open = "'"
                quote_close = "'"
        
        # Convert emphasis markers to quotation marks
        # First process bold **text** and __text__ (requires full pair)
        text = re.sub(r'\*\*([^*]+?)\*\*', rf'{quote_open}\1{quote_close}', text)
        text = re.sub(r'__([^_]+?)__', rf'{quote_open}\1{quote_close}', text)
        
        # Then process italics *text* and _text_ (avoid matching already processed bold)
        text = re.sub(r'(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)', rf'{quote_open}\1{quote_close}', text)
        text = re.sub(r'(?<!_)_(?!_)([^_\n]+?)(?<!_)_(?!_)', rf'{quote_open}\1{quote_close}', text)
        
        # Handle unpaired emphasis markers (may occur in streaming text)
        # Remove unpaired marker symbols, preserving content
        text = re.sub(r'\*\*([^*\n]+?)(?=\*\*|$)', r'\1', text)
        text = re.sub(r'__([^_\n]+?)(?=__|$)', r'\1', text)
        # Remove remaining unpaired markers
        text = re.sub(r'\*\*+', '', text)
        text = re.sub(r'__+', '', text)
        text = re.sub(r'(?<!\*)\*(?!\*)', '', text)  # Remove single unpaired *
        text = re.sub(r'(?<!_)_(?!_)', '', text)  # Remove single unpaired _
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Clean redundant whitespace.
        
        Strategy:
        - Merge multiple consecutive spaces into one
        - Replace newlines, tabs with space (TTS does not need newlines)
        - Clean multiple consecutive spaces
        - Do not strip leading/trailing spaces to preserve necessary spacing in streams
        """
        # Replace newlines, tabs with space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        # Merge multiple consecutive spaces into one
        text = re.sub(r' +', ' ', text)
        return text


def create_text_filter(
    remove_brackets: bool = True,
    remove_code_blocks: bool = True,
    remove_markdown_links: bool = True,
    remove_markdown_formatting: bool = True,
    convert_emphasis_to_quotes: bool = True,
    remove_special_tokens: bool = True,
    chinese_quote_style: str = '「」',
    english_quote_style: str = '"',
    chinese_threshold: float = 0.3,
) -> TextFilter:
    """
    Factory function to create a TextFilter.
    
    Args:
        remove_brackets: Whether to remove brackets and their contents.
        remove_code_blocks: Whether to remove code blocks.
        remove_markdown_links: Whether to remove markdown links and images.
        remove_markdown_formatting: Whether to remove markdown formatting.
        convert_emphasis_to_quotes: Whether to convert emphasis markers to quotes.
        remove_special_tokens: Whether to remove special tokens.
        chinese_quote_style: Chinese quote style.
        english_quote_style: English quote style.
        chinese_threshold: Threshold ratio of Chinese characters.
        
    Returns:
        TextFilter instance.
    """
    config = TextFilterConfig(
        remove_brackets=remove_brackets,
        remove_code_blocks=remove_code_blocks,
        remove_markdown_links=remove_markdown_links,
        remove_markdown_formatting=remove_markdown_formatting,
        convert_emphasis_to_quotes=convert_emphasis_to_quotes,
        remove_special_tokens=remove_special_tokens,
        chinese_quote_style=chinese_quote_style,
        english_quote_style=english_quote_style,
        chinese_threshold=chinese_threshold,
    )
    return TextFilter(config)
