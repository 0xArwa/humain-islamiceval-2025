"""
Verse Span Validator for Quranic text annotations.

This module provides validation functions to detect annotation issues in Quranic verse spans,
including non-Arabic characters, improper delimiters, and punctuation marks that shouldn't 
be part of Quranic text.
"""

import re
from typing import List, Dict, Set
import unicodedata


class VerseSpanValidator:
    """Validator for Quranic verse span annotations."""
    
    # Arabic Unicode ranges
    ARABIC_RANGES = [
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ]
    
    # Characters that shouldn't appear in Quranic text
    FORBIDDEN_CHARS = {
        '*',  # Asterisk (used as verse separator)
        '(',  # Opening parenthesis
        ')',  # Closing parenthesis
        '[',  # Opening square bracket
        ']',  # Closing square bracket
        '{',  # Opening curly brace
        '}',  # Closing curly brace
        '﴾',  # Ornate left parenthesis
        '﴿',  # Ornate right parenthesis
        '"',  # Quotation mark
        '"',  # Left double quotation mark
        '"',  # Right double quotation mark
        ''',  # Left single quotation mark
        ''',  # Right single quotation mark
        '«',  # Left-pointing double angle quotation mark
        '»',  # Right-pointing double angle quotation mark
    }
    
    # Punctuation that shouldn't end Quranic verses
    FORBIDDEN_END_PUNCTUATION = {
        '.',  # Period
        ',',  # Comma
        '،',  # Arabic comma
        ';',  # Semicolon
        '؛',  # Arabic semicolon
        ':',  # Colon
        '؞',  # Arabic triple dot punctuation mark
        '؍',  # Arabic date separator
        '!',  # Exclamation mark
        '؟',  # Arabic question mark
        '?',  # Question mark
        '﴾',  # Ornate left parenthesis
        '﴿',  # Ornate right parenthesis
        ')',  # Closing parenthesis
        ']',  # Closing square bracket
        '}',  # Closing curly brace
        '…',  # Horizontal ellipsis
        '...',  # Three dots (will be handled by the removal loop)
    }
    
    def __init__(self):
        """Initialize the validator."""
        pass
    
    def is_arabic_char(self, char: str) -> bool:
        """Check if a character is Arabic."""
        if not char:
            return False
        
        char_code = ord(char)
        for start, end in self.ARABIC_RANGES:
            if start <= char_code <= end:
                return True
        return False
    
    def is_arabic_or_space(self, char: str) -> bool:
        """Check if a character is Arabic or whitespace."""
        return self.is_arabic_char(char) or char.isspace()
    
    def _is_ayah_separator(self, text: str, pos: int) -> bool:
        """
        Check if character at position is part of an ayah separator that should be preserved.
        
        Ayah separators include:
        - * (asterisk)
        - Numbers in parentheses like (١٤) or (14)
        - Sequences of 3+ consecutive non-Arabic characters
        - Arabic comma (،) and regular comma (,) when used as verse separators
        
        Args:
            text (str): The full text
            pos (int): Position of current character
            
        Returns:
            bool: True if character should be preserved as part of ayah separator
        """
        char = text[pos]
        
        # Case 1: Asterisk is always an ayah separator
        if char == '*':
            return True
        
        # Case 2: Commas can be verse separators
        if char in '،,':
            return True
        
        # Case 3: Check for numbers in parentheses like (١٤) or (14)
        if char in '()':
            # Look for pattern like (number) around this position
            if self._is_parenthetical_number(text, pos):
                return True
        
        # Also check if it's a digit within parentheses
        if char.isdigit() or char in '٠١٢٣٤٥٦٧٨٩':
            # Check if surrounded by parentheses
            if self._is_parenthetical_number(text, pos):
                return True
        
        # Case 4: Check if this is part of a sequence of 3+ non-Arabic chars
        if not self.is_arabic_char(char) and not char.isspace():
            if self._is_part_of_non_arabic_sequence(text, pos):
                return True
        
        return False
    
    def _is_parenthetical_number(self, text: str, pos: int) -> bool:
        """Check if position is part of a parenthetical number like (١٤) or (14)."""
        import re
        
        # Look for pattern (digits) around the current position
        # Extend search window around current position
        start = max(0, pos - 10)
        end = min(len(text), pos + 10)
        substring = text[start:end]
        
        # Arabic numerals: ٠١٢٣٤٥٦٧٨٩
        # English numerals: 0123456789
        number_pattern = r'\([٠-٩0-9]+\)'
        
        for match in re.finditer(number_pattern, substring):
            match_start = start + match.start()
            match_end = start + match.end()
            if match_start <= pos < match_end:
                return True
        
        return False
    
    def _is_part_of_non_arabic_sequence(self, text: str, pos: int) -> bool:
        """Check if position is part of a sequence of 3+ consecutive non-Arabic characters."""
        char = text[pos]
        
        # Character must be non-Arabic and non-space
        if self.is_arabic_char(char) or char.isspace():
            return False
        
        # Count consecutive non-Arabic, non-space characters around this position
        count = 1
        
        # Count backward
        i = pos - 1
        while i >= 0 and not self.is_arabic_char(text[i]) and not text[i].isspace():
            count += 1
            i -= 1
        
        # Count forward
        i = pos + 1
        while i < len(text) and not self.is_arabic_char(text[i]) and not text[i].isspace():
            count += 1
            i += 1
        
        # Return True if sequence is 3 or more characters
        return count >= 3
    
    def validate_verse_span(self, verse_span: str) -> Dict[str, any]:
        """
        Validate a verse span and return detected issues.
        
        Args:
            verse_span (str): The verse span to validate
            
        Returns:
            Dict containing validation results and detected issues
        """
        if not verse_span or not verse_span.strip():
            return {
                'is_valid': False,
                'issues': ['empty_or_whitespace'],
                'details': {}
            }
        
        verse_span = verse_span.strip()
        issues = []
        details = {}
        
        # Check for non-Arabic characters at start
        if verse_span and not self.is_arabic_or_space(verse_span[0]):
            issues.append('starts_with_non_arabic')
            details['start_char'] = verse_span[0]
            details['start_char_name'] = unicodedata.name(verse_span[0], 'UNKNOWN')
        
        # Check for non-Arabic characters at end
        if verse_span and not self.is_arabic_or_space(verse_span[-1]):
            issues.append('ends_with_non_arabic')
            details['end_char'] = verse_span[-1]
            details['end_char_name'] = unicodedata.name(verse_span[-1], 'UNKNOWN')
        
        # Check for forbidden characters
        forbidden_found = set()
        for char in verse_span:
            if char in self.FORBIDDEN_CHARS:
                forbidden_found.add(char)
        
        if forbidden_found:
            issues.append('contains_forbidden_chars')
            details['forbidden_chars'] = list(forbidden_found)
        
        # Check for forbidden end punctuation
        if verse_span and verse_span[-1] in self.FORBIDDEN_END_PUNCTUATION:
            issues.append('ends_with_forbidden_punctuation')
            details['end_punctuation'] = verse_span[-1]
        
        # Check for asterisk (verse separator)
        if '*' in verse_span:
            issues.append('contains_verse_separator')
            details['asterisk_positions'] = [i for i, char in enumerate(verse_span) if char == '*']
        
        # Check for parentheses and brackets
        parentheses_brackets = {'(', ')', '[', ']', '{', '}', '﴾', '﴿'}
        found_delimiters = set()
        for char in verse_span:
            if char in parentheses_brackets:
                found_delimiters.add(char)
        
        if found_delimiters:
            issues.append('contains_delimiters')
            details['delimiters'] = list(found_delimiters)
        
        # Check for quotation marks
        quotes = {'"', '"', '"', ''', ''', '«', '»'}
        found_quotes = set()
        for char in verse_span:
            if char in quotes:
                found_quotes.add(char)
        
        if found_quotes:
            issues.append('contains_quotes')
            details['quotes'] = list(found_quotes)
        
        # Check for non-Arabic characters in the middle
        non_arabic_chars = []
        for i, char in enumerate(verse_span):
            if not self.is_arabic_or_space(char) and char not in self.FORBIDDEN_CHARS:
                non_arabic_chars.append({'char': char, 'position': i, 'name': unicodedata.name(char, 'UNKNOWN')})
        
        if non_arabic_chars:
            issues.append('contains_non_arabic_chars')
            details['non_arabic_chars'] = non_arabic_chars[:10]  # Limit to first 10
        
        # Check for excessive whitespace
        if '  ' in verse_span:  # Multiple consecutive spaces
            issues.append('excessive_whitespace')
        
        # Check for leading/trailing whitespace in original
        original_verse_span = verse_span  # We already stripped it
        if verse_span != verse_span.strip():
            issues.append('leading_trailing_whitespace')
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'details': details,
            'verse_span_length': len(verse_span),
            'arabic_char_count': sum(1 for char in verse_span if self.is_arabic_char(char))
        }
    
    def correct_verse_span(self, verse_span: str) -> Dict[str, any]:
        """
        Correct common issues in verse spans.
        
        Args:
            verse_span (str): The verse span to correct
            
        Returns:
            Dict containing corrected span and correction details
        """
        if not verse_span:
            return {
                'corrected_span': '',
                'corrections_applied': ['empty_input'],
                'original_span': verse_span
            }
        
        corrected = verse_span.strip()
        corrections_applied = []
        
        # Preserve ayah separators: *, parentheses with numbers, or sequences of 2+ non-Arabic chars
        # that serve as verse separators
        forbidden_chars_found = []
        cleaned = ''
        i = 0
        while i < len(corrected):
            char = corrected[i]
            
            # Check if this is an ayah separator pattern we should preserve
            if self._is_ayah_separator(corrected, i):
                # Keep the separator as is
                cleaned += char
                i += 1
            elif char in self.FORBIDDEN_CHARS:
                forbidden_chars_found.append(char)
                i += 1
            else:
                cleaned += char
                i += 1
        
        if forbidden_chars_found:
            corrected = cleaned
            corrections_applied.append(f"removed_forbidden_chars: {list(set(forbidden_chars_found))}")
        
        # Remove punctuation and non-Arabic characters from start and end
        # But preserve ayah separators
        import re
        
        # Continue cleaning until no more changes are needed
        changed = True
        while changed and corrected:
            changed = False
            original_corrected = corrected
            
            # Remove from start - but check if it's part of an ayah separator
            while corrected and (not self.is_arabic_or_space(corrected[0]) or corrected[0] in self.FORBIDDEN_END_PUNCTUATION):
                # Check if this character is part of an ayah separator
                if self._is_ayah_separator(corrected, 0):
                    break
                
                removed_char = corrected[0]
                corrected = corrected[1:].lstrip()
                if removed_char in self.FORBIDDEN_END_PUNCTUATION:
                    corrections_applied.append(f"removed_start_punctuation: {removed_char}")
                else:
                    corrections_applied.append(f"removed_start_non_arabic: {removed_char}")
                changed = True
            
            # Remove from end - but check if it's part of an ayah separator
            while corrected and (not self.is_arabic_or_space(corrected[-1]) or corrected[-1] in self.FORBIDDEN_END_PUNCTUATION):
                # Check if this character is part of an ayah separator
                if self._is_ayah_separator(corrected, len(corrected) - 1):
                    break
                
                removed_char = corrected[-1]
                corrected = corrected[:-1].rstrip()
                if removed_char in self.FORBIDDEN_END_PUNCTUATION:
                    corrections_applied.append(f"removed_end_punctuation: {removed_char}")
                else:
                    corrections_applied.append(f"removed_end_non_arabic: {removed_char}")
                changed = True
            
            # Special handling for ellipsis patterns (...) and similar sequences
            if re.search(r'\.{2,}$', corrected):
                corrected = re.sub(r'\.+$', '', corrected).rstrip()
                corrections_applied.append("removed_trailing_dots")
                changed = True
        
        # Clean up multiple consecutive spaces
        import re
        if re.search(r'\s{2,}', corrected):
            corrected = re.sub(r'\s+', ' ', corrected)
            corrections_applied.append("normalized_whitespace")
        
        # Final cleanup
        corrected = corrected.strip()
        
        return {
            'corrected_span': corrected,
            'corrections_applied': corrections_applied,
            'original_span': verse_span,
            'correction_needed': len(corrections_applied) > 0
        }
    
    def separate_verses(self, verse_span: str) -> List[str]:
        """
        Separate a verse span into individual verses based on ayah separators.
        
        Separators include:
        - * (asterisk)
        - Numbers in parentheses like (١٤) or (14) 
        - Sequences of 3+ consecutive non-Arabic characters
        - Arabic comma (،) as verse separator
        - Regular comma (,) as verse separator
        
        Args:
            verse_span (str): The verse span to separate
            
        Returns:
            List[str]: List of individual verses
        """
        if not verse_span or not verse_span.strip():
            return []
        
        import re
        
        # Clean up the span first
        cleaned_span = verse_span.strip()
        
        # Pattern to match various separators:
        # 1. Asterisk with optional surrounding spaces
        # 2. Numbers in parentheses (Arabic or English numerals)
        # 3. Sequences of 3+ non-Arabic, non-space characters (increased from 2+ to 3+ to be more conservative)
        # 4. Arabic comma with optional spaces
        # 5. Regular comma with optional spaces
        separator_patterns = [
            r'\s*\*\s*',  # Asterisk with optional spaces
            r'\s*\([٠-٩0-9]+\)\s*',  # Numbers in parentheses with optional spaces
            r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]{3,}',  # 3+ consecutive non-Arabic chars
            r'\s*،\s*',  # Arabic comma with optional spaces
            r'\s*,\s*',   # Regular comma with optional spaces
        ]
        
        # Combine all patterns with alternation
        combined_pattern = '|'.join(f'({pattern})' for pattern in separator_patterns)
        
        # Split the text using the combined pattern
        parts = re.split(combined_pattern, cleaned_span)
        
        # Filter out empty parts and separator matches
        verses = []
        for part in parts:
            if part and part.strip():
                # Check if this looks like a separator (contains only non-Arabic chars, spaces, and punctuation)
                cleaned_part = part.strip()
                if self._is_likely_separator(cleaned_part):
                    continue
                verses.append(cleaned_part.strip())
        
        return verses
    
    def _is_likely_separator(self, text: str) -> bool:
        """Check if text is likely a separator rather than actual verse content."""
        if not text:
            return True
        
        # Count Arabic characters
        arabic_count = sum(1 for char in text if self.is_arabic_char(char))
        total_chars = len([char for char in text if not char.isspace()])
        
        # If less than 50% Arabic characters, likely a separator
        if total_chars > 0 and arabic_count / total_chars < 0.5:
            return True
        
        # Check for specific separator patterns
        separator_patterns = [
            r'^\s*\*\s*$',  # Just asterisk
            r'^\s*\([٠-٩0-9]+\)\s*$',  # Just numbers in parentheses
            r'^[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]+$',  # Only non-Arabic chars
            r'^\s*،\s*$',  # Just Arabic comma
            r'^\s*,\s*$',   # Just regular comma
        ]
        
        import re
        for pattern in separator_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def get_issue_description(self, issue: str) -> str:
        """Get a human-readable description of an issue."""
        descriptions = {
            'empty_or_whitespace': 'Verse span is empty or contains only whitespace',
            'starts_with_non_arabic': 'Verse span starts with non-Arabic character',
            'ends_with_non_arabic': 'Verse span ends with non-Arabic character', 
            'contains_forbidden_chars': 'Contains forbidden characters (parentheses, brackets, quotes, etc.)',
            'ends_with_forbidden_punctuation': 'Ends with punctuation that should not be in Quranic text',
            'contains_verse_separator': 'Contains asterisk (*) which is used as verse separator',
            'contains_delimiters': 'Contains parentheses, brackets, or similar delimiters',
            'contains_quotes': 'Contains quotation marks',
            'contains_non_arabic_chars': 'Contains non-Arabic characters in the text',
            'excessive_whitespace': 'Contains excessive whitespace (multiple consecutive spaces)',
            'leading_trailing_whitespace': 'Contains leading or trailing whitespace'
        }
        return descriptions.get(issue, f'Unknown issue: {issue}')
    
    def validate_dataset(self, verse_spans: List[str]) -> Dict[str, any]:
        """
        Validate multiple verse spans and return summary statistics.
        
        Args:
            verse_spans (List[str]): List of verse spans to validate
            
        Returns:
            Dict containing validation summary and detailed results
        """
        results = []
        issue_counts = {}
        
        for i, verse_span in enumerate(verse_spans):
            validation_result = self.validate_verse_span(verse_span)
            validation_result['index'] = i
            validation_result['verse_span'] = verse_span[:100] + '...' if len(verse_span) > 100 else verse_span
            results.append(validation_result)
            
            # Count issues
            for issue in validation_result['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        total_spans = len(verse_spans)
        invalid_spans = sum(1 for result in results if not result['is_valid'])
        
        return {
            'total_spans': total_spans,
            'valid_spans': total_spans - invalid_spans,
            'invalid_spans': invalid_spans,
            'validation_rate': (total_spans - invalid_spans) / total_spans if total_spans > 0 else 0,
            'issue_counts': issue_counts,
            'detailed_results': results
        }


