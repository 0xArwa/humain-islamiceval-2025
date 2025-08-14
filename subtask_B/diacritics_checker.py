"""
Diacritics (Tashkeel) Checker for Islamic Text Verification

This module provides functions to validate Arabic diacritics (tashkeel) in Quranic verses.
It compares input verses with reference verses character by character to detect incorrect diacritics.
"""

import re
from typing import Tuple, Dict, Optional

# Arabic diacritics (tashkeel) characters
ARABIC_DIACRITICS = {
    'َ',   # Fatha
    'ُ',   # Damma
    'ِ',   # Kasra
    'ً',   # Fathatan
    'ٌ',   # Dammatan
    'ٍ',   # Kasratan
    'ْ',   # Sukun
    'ّ',   # Shadda
    'ٰ',   # Superscript Alif
    'ٔ',   # Hamza Above
}


def has_diacritics(text: str) -> bool:
    """
    Check if text contains Arabic diacritics.
    
    Args:
        text (str): Arabic text to check
        
    Returns:
        bool: True if text contains diacritics, False otherwise
    """
    return any(char in ARABIC_DIACRITICS for char in text)


def remove_diacritics(text: str) -> str:
    """
    Remove all diacritics from Arabic text.
    
    Args:
        text (str): Arabic text with diacritics
        
    Returns:
        str: Text without diacritics
    """
    result = ""
    for char in text:
        if char not in ARABIC_DIACRITICS:
            result += char
    return result


def align_text_for_diacritics_check(input_text: str, reference_text: str) -> Tuple[str, str]:
    """
    Align input and reference texts for diacritics comparison by removing non-Arabic characters
    and normalizing whitespace while preserving the core Arabic letters and diacritics.
    
    Args:
        input_text (str): Input text to check
        reference_text (str): Reference text to compare against
        
    Returns:
        Tuple[str, str]: Aligned input and reference texts
    """
    def clean_for_alignment(text):
        # Keep Arabic letters, diacritics, and spaces only
        arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0590-\u05FF\s]+'
        cleaned = ''.join(re.findall(arabic_pattern, text))
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip()
    
    aligned_input = clean_for_alignment(input_text)
    aligned_reference = clean_for_alignment(reference_text)
    
    return aligned_input, aligned_reference


def check_diacritics_mismatch(input_text: str, reference_text: str) -> Dict:
    """
    Check for diacritics mismatches between input and reference text.
    
    This function compares diacritics character by character only if the input text has diacritics.
    If input has no diacritics, it's considered valid (partial tashkeel is allowed).
    If input has diacritics, they must match the reference exactly.
    
    Args:
        input_text (str): Input verse with potential diacritics
        reference_text (str): Reference verse with correct diacritics
        
    Returns:
        Dict: Result with validation status and details
    """
    result = {
        'has_diacritics_mismatch': False,
        'input_has_diacritics': False,
        'reference_has_diacritics': False,
        'mismatch_positions': [],
        'base_text_matches': False,
        'diacritics_accuracy': 1.0,
        'details': ''
    }
    
    # Check if input has diacritics
    result['input_has_diacritics'] = has_diacritics(input_text)
    result['reference_has_diacritics'] = has_diacritics(reference_text)
    
    # If input has no diacritics, it's considered valid (partial tashkeel allowed)
    if not result['input_has_diacritics']:
        # Check if the base text (without diacritics) matches
        input_no_diacritics = remove_diacritics(input_text)
        reference_no_diacritics = remove_diacritics(reference_text)
        result['base_text_matches'] = input_no_diacritics.strip() == reference_no_diacritics.strip()
        result['details'] = 'Input has no diacritics - considered valid'
        return result
    
    # If input has diacritics, we need to check them against the reference
    if not result['reference_has_diacritics']:
        result['details'] = 'Input has diacritics but reference does not - cannot validate'
        return result
    
    # Align texts for comparison
    aligned_input, aligned_reference = align_text_for_diacritics_check(input_text, reference_text)
    
    # Check if base texts match (without diacritics)
    input_base = remove_diacritics(aligned_input)
    reference_base = remove_diacritics(aligned_reference)
    result['base_text_matches'] = input_base == reference_base
    
    # If base texts don't match, we can't do meaningful diacritics comparison
    if not result['base_text_matches']:
        result['details'] = 'Base text mismatch - cannot compare diacritics'
        return result
    
    # Compare diacritics character by character
    mismatches = []
    total_diacritics_positions = 0
    correct_diacritics = 0
    
    min_len = min(len(aligned_input), len(aligned_reference))
    
    for i in range(min_len):
        input_char = aligned_input[i] if i < len(aligned_input) else ''
        ref_char = aligned_reference[i] if i < len(aligned_reference) else ''
        
        # If both are diacritics, compare them
        if input_char in ARABIC_DIACRITICS or ref_char in ARABIC_DIACRITICS:
            total_diacritics_positions += 1
            if input_char != ref_char:
                mismatches.append({
                    'position': i,
                    'input_char': input_char,
                    'reference_char': ref_char,
                    'context': aligned_input[max(0, i-5):i+6]
                })
            else:
                correct_diacritics += 1
    
    # Calculate accuracy
    if total_diacritics_positions > 0:
        result['diacritics_accuracy'] = correct_diacritics / total_diacritics_positions
    
    # Determine if there's a mismatch
    result['has_diacritics_mismatch'] = len(mismatches) > 0
    result['mismatch_positions'] = mismatches
    
    if result['has_diacritics_mismatch']:
        result['details'] = f'Found {len(mismatches)} diacritics mismatches out of {total_diacritics_positions} positions'
    else:
        result['details'] = f'All {total_diacritics_positions} diacritics match correctly'
    
    return result


def should_reject_for_diacritics_mismatch(
    input_text: str, 
    reference_text: str, 
    mismatch_threshold: float = 0.8
) -> Tuple[bool, Dict]:
    """
    Determine if a verse should be rejected due to diacritics mismatches.
    
    Args:
        input_text (str): Input verse to check
        reference_text (str): Reference verse to compare against
        mismatch_threshold (float): Minimum accuracy required (default: 0.8)
        
    Returns:
        Tuple[bool, Dict]: (should_reject, detailed_results)
    """
    # Check diacritics
    diacritics_result = check_diacritics_mismatch(input_text, reference_text)
    
    should_reject = False
    
    # Only reject if:
    # 1. Input has diacritics AND
    # 2. There are diacritics mismatches AND
    # 3. Accuracy is below threshold
    if (diacritics_result['input_has_diacritics'] and 
        diacritics_result['has_diacritics_mismatch'] and 
        diacritics_result['diacritics_accuracy'] < mismatch_threshold):
        should_reject = True
    
    return should_reject, diacritics_result