"""
Quran Verse Verification System for Subtask 1B

This is an optimized version that uses indexing for faster lookups.
"""

import json
from typing import List, Dict, Set
from difflib import SequenceMatcher
from collections import defaultdict
from diacritics_checker import should_reject_for_diacritics_mismatch


USE_MULTIWORD_SUBSTRING_LOGIC = True


def normalize_ayah(text: str) -> str:
    """
    Normalize Ayah text for comparison.
    
    Args:
        text (str): Input Ayah text
        
    Returns:
        str: Normalized text
    """
    import re
    
    if not text:
        return ""
        
    # Remove common punctuation and diacritics
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    text = re.sub(r'[،؛:.!?()«»""'']', ' ', text)  # Remove punctuation
    text = re.sub(r'[0-9\[\]\/]', ' ', text)  # Remove numbers and brackets
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    
    return text.lower()
USE_DIACRITICS_CHECKING = True

class QuranVerifier:
    """
    Optimized Quran verifier using text indexing for faster lookups.
    """
    
    def __init__(self, quranic_verses_path: str, multiverse_strategy: str = 'any'):
        """
        Initialize the verifier with indexed Quranic reference data.
        
        Args:
            quranic_verses_path (str): Path to quranic_verses.json file
            multiverse_strategy (str): Strategy for multi-verse validation:
                - 'any': At least one verse matches (lenient, current default)
                - 'majority': More than 50% of verses match 
                - 'all': All verses must match (strictest)
                - 'high': At least 80% of verses must match
        """
        self.reference_verses = self._load_quranic_verses(quranic_verses_path)
        self.normalized_verses = self._create_normalized_index()
        self.word_index = self._create_word_index()
        self.multiverse_strategy = multiverse_strategy
        
    def _load_quranic_verses(self, path: str) -> List[Dict]:
        """Load Quranic verses from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_normalized_index(self) -> Dict[str, Dict]:
        """Create an index of normalized verses for fast lookup."""
        normalized_index = {}
        for verse_data in self.reference_verses:
            normalized_text = normalize_ayah(verse_data['ayah_text'])
            normalized_index[normalized_text] = verse_data
        return normalized_index
    
    def _create_word_index(self) -> Dict[str, List[Dict]]:
        """Create a word-based index for faster partial matching."""
        word_index = defaultdict(list)
        for verse_data in self.reference_verses:
            normalized_text = normalize_ayah(verse_data['ayah_text'])
            words = normalized_text.split()
            for word in words:
                if len(word) >= 3:  # Only index words with 3+ characters
                    word_index[word].append(verse_data)
        return dict(word_index)
    
    def _get_candidate_verses(self, input_verse: str) -> Set[Dict]:
        """Get candidate verses based on word overlap."""
        normalized_input = normalize_ayah(input_verse)
        words = normalized_input.split()
        
        candidates = set()
        for word in words:
            if len(word) >= 3 and word in self.word_index:
                for verse_data in self.word_index[word]:
                    candidates.add(id(verse_data))  # Use object id as key
        
        # Convert back to verse data objects
        result_candidates = []
        for verse_data in self.reference_verses:
            if id(verse_data) in candidates:
                result_candidates.append(verse_data)
        
        return result_candidates
    
    def verify_verse(self, input_verse: str, threshold: float = 0.95) -> Dict:
        """
        Verify if an input verse matches any reference Quranic verse (optimized).
        
        Args:
            input_verse (str): Input verse to verify
            threshold (float): Similarity threshold (default: 0.95)
            
        Returns:
            Dict: Verification result with match status and details
        """
        normalized_input = normalize_ayah(input_verse)
        
        best_match = {
            'is_match': False,
            'similarity': 0.0,
            'matched_verse': None,
            'surah_name': '',
            'ayah_id': 0,
            'normalized_input': normalized_input,
            'normalized_reference': ''
        }
        
        # Quick exact match check
        if normalized_input in self.normalized_verses:
            verse_data = self.normalized_verses[normalized_input]
            best_match.update({
                'is_match': True,
                'similarity': 1.0,
                'matched_verse': verse_data['ayah_text'],
                'surah_name': verse_data['surah_name'],
                'ayah_id': verse_data['ayah_id'],
                'normalized_reference': normalized_input
            })
            return best_match
        
        # Get candidate verses for partial matching
        candidates = self._get_candidate_verses(input_verse)
        
        # Check candidates for substring and similarity matches
        for verse_data in candidates:
            reference_text = verse_data['ayah_text']
            normalized_reference = normalize_ayah(reference_text)
            
            # Check if input is substring of reference
            is_substring = normalized_input in normalized_reference
            
            # Calculate similarity
            similarity = SequenceMatcher(None, normalized_input, normalized_reference).ratio()
            
            # NEW LOGIC: Multi-word substring matching (can be reverted via USE_MULTIWORD_SUBSTRING_LOGIC flag)
            should_match = False
            if USE_MULTIWORD_SUBSTRING_LOGIC and is_substring:
                # Count words in input for multi-word substring logic
                input_words = normalized_input.split()
                if len(input_words) > 2:
                    # Multi-word substring match - accept this even with low similarity
                    should_match = True
                    # Only boost similarity slightly to ensure proper ranking, don't over-boost
                    similarity = max(similarity, 0.85)
                else:
                    # Use original logic for 1-2 word inputs
                    should_match = (is_substring or similarity >= threshold) and similarity > best_match['similarity']
            else:
                # Original logic (used when flag is False or no substring match)
                should_match = (is_substring or similarity >= threshold) and similarity > best_match['similarity']
            
            if should_match:
                # Apply diacritics checking if enabled
                final_match = True
                diacritics_result = {}
                
                if USE_DIACRITICS_CHECKING:
                    should_reject, diacritics_result = should_reject_for_diacritics_mismatch(
                        input_verse, reference_text, mismatch_threshold=0.8
                    )
                    if should_reject:
                        final_match = False
                
                if final_match:
                    best_match.update({
                        'is_match': True,
                        'similarity': similarity,
                        'matched_verse': reference_text,
                        'surah_name': verse_data['surah_name'],
                        'ayah_id': verse_data['ayah_id'],
                        'normalized_reference': normalized_reference,
                        'diacritics_check': diacritics_result if USE_DIACRITICS_CHECKING else None
                    })
        
        return best_match
    
    def verify_separated_verses(self, input_verse: str, threshold: float = 0.95) -> Dict:
        """
        Verify a verse span by first separating it into individual verses, then checking each verse.
        
        Args:
            input_verse (str): Input verse span (may contain multiple verses)
            threshold (float): Similarity threshold (default: 0.95)
            
        Returns:
            Dict: Verification result with details about separated verses
        """
        from verse_span_validator import VerseSpanValidator
        
        validator = VerseSpanValidator()
        separated_verses = validator.separate_verses(input_verse)
        
        results = {
            'original_input': input_verse,
            'separated_verses': separated_verses,
            'verse_count': len(separated_verses),
            'verification_results': [],
            'overall_match': False,
            'matched_verse_count': 0,
            'best_overall_similarity': 0.0
        }
        
        # Verify each separated verse
        for i, verse in enumerate(separated_verses):
            verse_result = self.verify_verse(verse, threshold)
            verse_result['verse_index'] = i
            verse_result['separated_verse'] = verse
            results['verification_results'].append(verse_result)
            
            if verse_result['is_match']:
                results['matched_verse_count'] += 1
            
            # Track best similarity across all verses
            if verse_result['similarity'] > results['best_overall_similarity']:
                results['best_overall_similarity'] = verse_result['similarity']
        
        # Consider overall match based on configurable strategy
        results['overall_match'] = self._evaluate_multiverse_match(results)
        
        return results
    
    def verify_verse_strict_substring(self, input_verse: str) -> Dict:
        """
        Verify if an input verse matches any reference Quranic verse using STRICT substring matching.
        
        Unlike the regular verify_verse method which uses similarity scoring, this method
        only returns a match if the normalized input verse is an exact substring of a 
        normalized reference verse.
        
        This function can be easily switched back to the original logic if needed.
        
        Args:
            input_verse (str): Input verse to verify
            
        Returns:
            Dict: Verification result with match status and details
        """
        normalized_input = normalize_ayah(input_verse)
        
        best_match = {
            'is_match': False,
            'similarity': 0.0,
            'matched_verse': None,
            'surah_name': '',
            'ayah_id': 0,
            'normalized_input': normalized_input,
            'normalized_reference': '',
            'match_type': 'strict_substring'
        }
        
        # Quick exact match check first
        if normalized_input in self.normalized_verses:
            verse_data = self.normalized_verses[normalized_input]
            best_match.update({
                'is_match': True,
                'similarity': 1.0,
                'matched_verse': verse_data['ayah_text'],
                'surah_name': verse_data['surah_name'],
                'ayah_id': verse_data['ayah_id'],
                'normalized_reference': normalized_input,
                'match_type': 'exact_match'
            })
            return best_match
        
        # Check all reference verses for substring matches
        for verse_data in self.reference_verses:
            reference_text = verse_data['ayah_text']
            normalized_reference = normalize_ayah(reference_text)
            
            # Strict substring check: input must be substring of reference
            if normalized_input in normalized_reference:
                # Calculate similarity for informational purposes
                similarity = SequenceMatcher(None, normalized_input, normalized_reference).ratio()
                
                # Update best match (prefer longer references that contain the input)
                if similarity > best_match['similarity']:
                    best_match.update({
                        'is_match': True,
                        'similarity': similarity,
                        'matched_verse': reference_text,
                        'surah_name': verse_data['surah_name'],
                        'ayah_id': verse_data['ayah_id'],
                        'normalized_reference': normalized_reference,
                        'match_type': 'substring_match'
                    })
        
        return best_match
    
    def verify_separated_verses_strict_substring(self, input_verse: str) -> Dict:
        """
        Verify a verse span using STRICT substring matching by first separating it into individual verses.
        
        Args:
            input_verse (str): Input verse span (may contain multiple verses)
            
        Returns:
            Dict: Verification result with details about separated verses
        """
        from verse_span_validator import VerseSpanValidator
        
        validator = VerseSpanValidator()
        separated_verses = validator.separate_verses(input_verse)
        
        results = {
            'original_input': input_verse,
            'separated_verses': separated_verses,
            'verse_count': len(separated_verses),
            'verification_results': [],
            'overall_match': False,
            'matched_verse_count': 0,
            'best_overall_similarity': 0.0,
            'match_type': 'strict_substring_multi'
        }
        
        # Verify each separated verse using strict substring matching
        for i, verse in enumerate(separated_verses):
            verse_result = self.verify_verse_strict_substring(verse)
            verse_result['verse_index'] = i
            verse_result['separated_verse'] = verse
            results['verification_results'].append(verse_result)
            
            if verse_result['is_match']:
                results['matched_verse_count'] += 1
            
            # Track best similarity across all verses
            if verse_result['similarity'] > results['best_overall_similarity']:
                results['best_overall_similarity'] = verse_result['similarity']
        
        # Consider overall match based on configurable strategy
        results['overall_match'] = self._evaluate_multiverse_match(results)
        
        return results
    
    def _evaluate_multiverse_match(self, results: Dict) -> bool:
        """
        Evaluate whether a multi-verse span should be considered a match
        based on the configured strategy.
        
        Args:
            results (Dict): Results from verify_separated_verses
            
        Returns:
            bool: True if the span should be considered a match
        """
        matched_count = results['matched_verse_count']
        total_count = results['verse_count']
        
        if total_count == 0:
            return False
        
        match_rate = matched_count / total_count
        
        if self.multiverse_strategy == 'any':
            # At least one verse matches (original behavior)
            return matched_count > 0
        elif self.multiverse_strategy == 'all':
            # All verses must match
            return matched_count == total_count
        elif self.multiverse_strategy == 'majority':
            # More than 50% must match
            return match_rate > 0.5
        elif self.multiverse_strategy == 'high':
            # At least 80% must match
            return match_rate >= 0.8
        else:
            # Default to 'any' strategy
            return matched_count > 0