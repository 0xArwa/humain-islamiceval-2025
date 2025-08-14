"""
Multi-verse detection that reduces false positives.
"""

import re
from typing import List, Dict


def is_multi_verse_span(verse_span: str) -> bool:
    """
    Detection of multi-verse spans that reduces false positives.

    Uses more conservative rules:
    - Asterisks (*) are always separators
    - Parenthetical verse numbers are always separators  
    - 3+ consecutive non-Arabic characters are separators
    - Commas are only separators under specific conditions
    
    Args:
        verse_span (str): The verse span to analyze
        
    Returns:
        bool: True if span likely contains multiple verses
    """
    if not verse_span or not verse_span.strip():
        return False
    
    # Arabic Unicode ranges for detecting non-Arabic characters
    arabic_ranges = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]
    
    def is_arabic_char(char: str) -> bool:
        """Check if a character is Arabic."""
        if not char:
            return False
        char_code = ord(char)
        for start, end in arabic_ranges:
            if start <= char_code <= end:
                return True
        return False
    
    # Check for clear separators (always indicate multi-verse)
    clear_separators = [
        r'\*',  # Asterisk
        r'\([٠-٩0-9]+\)',  # Numbers in parentheses
    ]
    
    for pattern in clear_separators:
        if re.search(pattern, verse_span):
            return True
    
    # Check for sequences of 3+ consecutive non-Arabic characters
    non_arabic_sequence = ''
    for char in verse_span:
        if not is_arabic_char(char) and not char.isspace():
            non_arabic_sequence += char
        else:
            if len(non_arabic_sequence) >= 3:
                return True
            non_arabic_sequence = ''
    
    # Check final sequence
    if len(non_arabic_sequence) >= 3:
        return True
    
    # Smart comma detection - only treat as separator if:
    # 1. Multiple commas (more likely to be verse separators)
    # 2. Comma near end of text (often verse separator)
    # 3. Comma followed by و (waw) which often starts new verses
    # 4. Long text with comma in middle (more likely to be multi-verse)
    
    comma_count = verse_span.count('،') + verse_span.count(',')
    
    if comma_count > 0:
        # Rule 1: Multiple commas - more likely separators
        if comma_count >= 2:
            return True
        
        # Rule 2: Single comma near end (last 20% of text)
        text_length = len(verse_span)
        for match in re.finditer(r'[،,]', verse_span):
            comma_pos = match.start()
            if comma_pos > text_length * 0.8:  # In last 20% of text
                return True
        
        # Rule 3: Comma followed by و (waw) - common verse starting pattern
        if re.search(r'[،,]\s*و', verse_span):
            # Additional check: make sure there's substantial text after the comma
            for match in re.finditer(r'[،,]\s*و', verse_span):
                remaining_text = verse_span[match.end():]
                # If there's significant text (>15 chars) after comma+waw, likely multi-verse
                if len(remaining_text.strip()) > 15:
                    return True
        
        # Rule 4: Long text with comma - more likely to be multi-verse
        if len(verse_span) > 200 and comma_count == 1:
            # Check if comma is roughly in the middle (not at very beginning or end)
            for match in re.finditer(r'[،,]', verse_span):
                comma_pos = match.start()
                relative_pos = comma_pos / text_length
                if 0.2 < relative_pos < 0.8:  # Comma in middle 60% of text
                    return True
    
    return False


def get_separator_info(verse_span: str) -> dict:
    """
    Analyze the types of separators found in a verse span with smart detection.
    
    Args:
        verse_span (str): The verse span to analyze
        
    Returns:
        dict: Information about detected separators
    """
    if not verse_span:
        return {'has_separators': False, 'separator_types': []}
    
    separator_info = {
        'has_separators': False,
        'separator_types': [],
        'separator_details': {}
    }
    
    # Check for different separator types
    if '*' in verse_span:
        separator_info['has_separators'] = True
        separator_info['separator_types'].append('asterisk')
        separator_info['separator_details']['asterisk_count'] = verse_span.count('*')
    
    # Check for parenthetical numbers
    parenthetical_matches = re.findall(r'\([٠-٩0-9]+\)', verse_span)
    if parenthetical_matches:
        separator_info['has_separators'] = True
        separator_info['separator_types'].append('parenthetical_numbers')
        separator_info['separator_details']['parenthetical_numbers'] = parenthetical_matches
    
    # Smart comma detection
    comma_count = verse_span.count('،') + verse_span.count(',')
    if comma_count > 0:
        # Apply the same smart rules as in is_multi_verse_span_smart
        is_comma_separator = False
        
        if comma_count >= 2:
            is_comma_separator = True
            separator_info['separator_details']['comma_reason'] = 'multiple_commas'
        elif len(verse_span) > 200:
            text_length = len(verse_span)
            for match in re.finditer(r'[،,]', verse_span):
                comma_pos = match.start()
                relative_pos = comma_pos / text_length
                if 0.2 < relative_pos < 0.8:
                    is_comma_separator = True
                    separator_info['separator_details']['comma_reason'] = 'long_text_middle_comma'
                    break
                elif comma_pos > text_length * 0.8:
                    is_comma_separator = True
                    separator_info['separator_details']['comma_reason'] = 'comma_near_end'
                    break
        elif re.search(r'[،,]\s*و', verse_span):
            for match in re.finditer(r'[،,]\s*و', verse_span):
                remaining_text = verse_span[match.end():]
                if len(remaining_text.strip()) > 15:
                    is_comma_separator = True
                    separator_info['separator_details']['comma_reason'] = 'comma_followed_by_waw'
                    break
        
        if is_comma_separator:
            separator_info['has_separators'] = True
            if '،' in verse_span:
                separator_info['separator_types'].append('arabic_comma_smart')
                separator_info['separator_details']['arabic_comma_count'] = verse_span.count('،')
            if ',' in verse_span:
                separator_info['separator_types'].append('regular_comma_smart')
                separator_info['separator_details']['regular_comma_count'] = verse_span.count(',')
    
    # Check for non-Arabic sequences
    arabic_ranges = [
        (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)
    ]
    
    def is_arabic_char(char: str) -> bool:
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in arabic_ranges)
    
    non_arabic_sequences = []
    current_sequence = ''
    for char in verse_span:
        if not is_arabic_char(char) and not char.isspace():
            current_sequence += char
        else:
            if len(current_sequence) >= 3:
                non_arabic_sequences.append(current_sequence)
            current_sequence = ''
    
    # Check final sequence
    if len(current_sequence) >= 3:
        non_arabic_sequences.append(current_sequence)
    
    if non_arabic_sequences:
        separator_info['has_separators'] = True
        separator_info['separator_types'].append('non_arabic_sequences')
        separator_info['separator_details']['non_arabic_sequences'] = non_arabic_sequences
    
    return separator_info
