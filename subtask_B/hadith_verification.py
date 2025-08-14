"""
Hadith Verification System for Subtask 1B

This module provides verification for Hadith texts against the six Hadith books reference.
"""

import json
import re
from typing import List, Dict, Set
from difflib import SequenceMatcher
from collections import defaultdict


def normalize_hadith(text: str) -> str:
    """
    Normalize Hadith text for comparison.
    
    Args:
        text (str): Input Hadith text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
        
    # Remove common punctuation and diacritics
    text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    text = re.sub(r'[،؛:.!?()«»""'']', ' ', text)  # Remove punctuation
    text = re.sub(r'[0-9\[\]\/]', ' ', text)  # Remove numbers and brackets
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    
    # Convert some characters for consistency
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    
    return text.lower()


class HadithVerifier:
    """
    Hadith verifier using text indexing for faster lookups against six Hadith books.
    """
    
    def __init__(self, hadith_books_path: str, multiverse_strategy: str = 'any'):
        """
        Initialize the verifier with indexed Hadith reference data.
        
        Args:
            hadith_books_path (str): Path to six_hadith_books.json file
            multiverse_strategy (str): Strategy for multi-hadith validation:
                - 'any': At least one hadith matches (lenient, current default)
                - 'majority': More than 50% of hadiths match 
                - 'all': All hadiths must match (strictest)
                - 'high': At least 80% of hadiths must match
        """
        self.reference_hadiths = self._load_hadith_books(hadith_books_path)
        self.normalized_hadiths = self._create_normalized_index()
        self.word_index = self._create_word_index()
        self.multiverse_strategy = multiverse_strategy
        
    def _load_hadith_books(self, path: str) -> List[Dict]:
        """Load Hadith books from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_normalized_index(self) -> Dict[str, Dict]:
        """Create an index of normalized hadiths for fast lookup."""
        normalized_index = {}
        for hadith_data in self.reference_hadiths:
            # Use Matn if available, otherwise hadithTxt
            hadith_text = hadith_data.get('Matn') or hadith_data.get('hadithTxt', '')
            if hadith_text:
                normalized_text = normalize_hadith(hadith_text)
                if normalized_text:  # Only add non-empty normalized text
                    normalized_index[normalized_text] = hadith_data
        return normalized_index
    
    def _create_word_index(self) -> Dict[str, List[Dict]]:
        """Create a word-based index for faster partial matching."""
        word_index = defaultdict(list)
        for hadith_data in self.reference_hadiths:
            # Use Matn if available, otherwise hadithTxt
            hadith_text = hadith_data.get('Matn') or hadith_data.get('hadithTxt', '')
            if hadith_text:
                normalized_text = normalize_hadith(hadith_text)
                words = normalized_text.split()
                for word in words:
                    if len(word) >= 3:  # Only index words with 3+ characters
                        word_index[word].append(hadith_data)
        return dict(word_index)
    
    def _get_candidate_hadiths(self, input_hadith: str) -> List[Dict]:
        """Get candidate hadiths based on word overlap with performance optimization."""
        normalized_input = normalize_hadith(input_hadith)
        words = normalized_input.split()
        
        candidates = set()
        word_count = 0
        
        # For very long inputs, be more selective with words to avoid huge candidate sets
        max_words = 15 if len(words) > 20 else len(words)
        
        for word in words[:max_words]:  # Limit word processing for long texts
            if len(word) >= 3 and word in self.word_index:
                for hadith_data in self.word_index[word]:
                    candidates.add(id(hadith_data))  # Use object id as key
                word_count += 1
                
                # Limit candidate set size to prevent performance degradation
                if len(candidates) > 1000:
                    break
        
        # Convert back to hadith data objects
        result_candidates = []
        for hadith_data in self.reference_hadiths:
            if id(hadith_data) in candidates:
                result_candidates.append(hadith_data)
                
                # Hard limit on candidates returned
                if len(result_candidates) >= 500:
                    break
        
        return result_candidates
    
    def verify_hadith(self, input_hadith: str, threshold: float = 0.95) -> Dict:
        """
        Verify if an input hadith matches any reference Hadith with intelligent matching.
        
        Args:
            input_hadith (str): Input hadith to verify
            threshold (float): Similarity threshold (default: 0.95)
            
        Returns:
            Dict: Verification result with match status and details
        """
        normalized_input = normalize_hadith(input_hadith)
        
        best_match = {
            'is_match': False,
            'similarity': 0.0,
            'matched_hadith': None,
            'hadith_id': 0,
            'book_id': 0,
            'title': '',
            'normalized_input': normalized_input,
            'normalized_reference': '',
            'match_type': 'none'
        }
        
        # First check for exact match in normalized index
        if normalized_input in self.normalized_hadiths:
            matched_hadith = self.normalized_hadiths[normalized_input]
            best_match.update({
                'is_match': True,
                'similarity': 1.0,
                'matched_hadith': matched_hadith.get('Matn') or matched_hadith.get('hadithTxt', ''),
                'hadith_id': matched_hadith.get('hadithID', 0),
                'book_id': matched_hadith.get('BookID', 0),
                'title': matched_hadith.get('title', ''),
                'normalized_reference': normalized_input,
                'match_type': 'exact'
            })
            return best_match
        
        # Determine input characteristics for intelligent matching
        input_word_count = len(normalized_input.split())
        is_short_input = input_word_count <= 8  # Consider 8 words or less as short
        
        # Get candidates for checking
        candidates = self._get_candidate_hadiths(input_hadith)
        
        # Try different matching strategies based on input length
        for hadith_data in candidates:
            hadith_text = hadith_data.get('Matn') or hadith_data.get('hadithTxt', '')
            if not hadith_text:
                continue
                
            normalized_reference = normalize_hadith(hadith_text)
            if not normalized_reference:
                continue
                
            ref_word_count = len(normalized_reference.split())
            
            # Strategy 1: Substring matching for short inputs
            if is_short_input and normalized_input in normalized_reference:
                # Calculate contextual similarity for substring matches
                substring_similarity = self._calculate_substring_similarity(
                    normalized_input, normalized_reference
                )
                
                if substring_similarity > best_match['similarity']:
                    best_match.update({
                        'similarity': substring_similarity,
                        'matched_hadith': hadith_text,
                        'hadith_id': hadith_data.get('hadithID', 0),
                        'book_id': hadith_data.get('BookID', 0),
                        'title': hadith_data.get('title', ''),
                        'normalized_reference': normalized_reference,
                        'match_type': 'substring'
                    })
            
            # Strategy 2: Fuzzy substring for short inputs with slight variations
            elif is_short_input:
                fuzzy_similarity = self._calculate_fuzzy_substring_similarity(
                    normalized_input, normalized_reference
                )
                
                if fuzzy_similarity > best_match['similarity']:
                    best_match.update({
                        'similarity': fuzzy_similarity,
                        'matched_hadith': hadith_text,
                        'hadith_id': hadith_data.get('hadithID', 0),
                        'book_id': hadith_data.get('BookID', 0),
                        'title': hadith_data.get('title', ''),
                        'normalized_reference': normalized_reference,
                        'match_type': 'fuzzy_substring'
                    })
            
            # Strategy 3: Full similarity matching for longer inputs
            else:
                similarity = SequenceMatcher(None, normalized_input, normalized_reference).ratio()
                
                if similarity > best_match['similarity']:
                    best_match.update({
                        'similarity': similarity,
                        'matched_hadith': hadith_text,
                        'hadith_id': hadith_data.get('hadithID', 0),
                        'book_id': hadith_data.get('BookID', 0),
                        'title': hadith_data.get('title', ''),
                        'normalized_reference': normalized_reference,
                        'match_type': 'full_similarity'
                    })
        
        # Determine if it's a match based on adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold(
            threshold, input_word_count, best_match['match_type']
        )
        best_match['is_match'] = best_match['similarity'] >= adaptive_threshold
        
        return best_match
    
    def verify_hadith_strict_substring(self, input_hadith: str) -> Dict:
        """
        Verify hadith using strict substring matching.
        
        Args:
            input_hadith (str): Input hadith text
            
        Returns:
            Dict: Verification result
        """
        normalized_input = normalize_hadith(input_hadith)
        
        best_match = {
            'is_match': False,
            'similarity': 0.0,
            'matched_hadith': None,
            'hadith_id': 0,
            'book_id': 0,
            'title': '',
            'normalized_input': normalized_input,
            'normalized_reference': ''
        }
        
        # Check all hadiths for substring match
        for hadith_data in self.reference_hadiths:
            hadith_text = hadith_data.get('Matn') or hadith_data.get('hadithTxt', '')
            if not hadith_text:
                continue
                
            normalized_reference = normalize_hadith(hadith_text)
            if not normalized_reference:
                continue
            
            # Check if input is a substring of reference
            if normalized_input in normalized_reference:
                # Calculate how much of the reference is covered
                coverage = len(normalized_input) / len(normalized_reference)
                
                if coverage > best_match['similarity']:
                    best_match.update({
                        'is_match': True,
                        'similarity': coverage,
                        'matched_hadith': hadith_text,
                        'hadith_id': hadith_data.get('hadithID', 0),
                        'book_id': hadith_data.get('BookID', 0),
                        'title': hadith_data.get('title', ''),
                        'normalized_reference': normalized_reference
                    })
        
        return best_match
    
    def verify_separated_hadiths(self, input_hadith: str, threshold: float = 0.95) -> Dict:
        """
        Verify multiple hadiths separated by common separators.
        
        Args:
            input_hadith (str): Input containing potentially multiple hadiths
            threshold (float): Similarity threshold
            
        Returns:
            Dict: Verification results for all hadiths
        """
        # Split on common separators
        separators = r'[*،؛\n]|(?:\s*\(\d+\)\s*)|(?:\s+و\s+)|(?:\s+ثم\s+)'
        hadith_parts = re.split(separators, input_hadith)
        hadith_parts = [part.strip() for part in hadith_parts if part.strip()]
        
        results = {
            'input_hadith': input_hadith,
            'separated_hadiths': hadith_parts,
            'verification_results': [],
            'overall_match': False,
            'matched_count': 0,
            'total_count': len(hadith_parts),
            'best_overall_similarity': 0.0
        }
        
        for hadith_part in hadith_parts:
            if len(hadith_part.strip()) < 5:  # Skip very short parts
                continue
                
            verification_result = self.verify_hadith(hadith_part, threshold)
            results['verification_results'].append(verification_result)
            
            if verification_result['is_match']:
                results['matched_count'] += 1
                
            # Update best overall similarity
            if verification_result['similarity'] > results['best_overall_similarity']:
                results['best_overall_similarity'] = verification_result['similarity']
        
        # Determine overall match based on strategy
        results['overall_match'] = self._evaluate_multiverse_match(results)
        
        return results
    
    def _evaluate_multiverse_match(self, results: Dict) -> bool:
        """
        Evaluate if a multi-hadith span should be considered a match based on strategy.
        
        Args:
            results (Dict): Results containing matched_count and total_count
            
        Returns:
            bool: True if considered a match based on strategy
        """
        if results['total_count'] == 0:
            return False
    
    def _calculate_substring_similarity(self, input_text: str, reference_text: str) -> float:
        """
        Calculate similarity score for substring matches with context awareness.
        
        Args:
            input_text (str): Normalized input text
            reference_text (str): Normalized reference text
            
        Returns:
            float: Contextual similarity score (0.6-1.0 range for valid substrings)
        """
        if not input_text or not reference_text or input_text not in reference_text:
            return 0.0
            
        # Base score for valid substring
        base_score = 0.75
        
        # Bonus for longer matches
        input_length = len(input_text)
        length_bonus = min(0.2, input_length / 200)  # Up to 0.2 bonus for longer texts
        
        # Bonus for word boundary matches (cleaner matches)
        input_words = input_text.split()
        ref_words = reference_text.split()
        
        # Check if input forms complete word boundaries in reference
        ref_text_joined = ' '.join(ref_words)
        if f' {input_text} ' in f' {ref_text_joined} ' or ref_text_joined.startswith(input_text) or ref_text_joined.endswith(input_text):
            boundary_bonus = 0.1
        else:
            boundary_bonus = 0.0
            
        # Calculate final score
        final_score = min(1.0, base_score + length_bonus + boundary_bonus)
        return final_score
    
    def _calculate_fuzzy_substring_similarity(self, input_text: str, reference_text: str) -> float:
        """
        Fast fuzzy substring similarity using optimized approach.
        
        Args:
            input_text (str): Normalized input text
            reference_text (str): Normalized reference text
            
        Returns:
            float: Fuzzy similarity score
        """
        if not input_text or not reference_text:
            return 0.0
            
        # Quick exit if input is too long for fuzzy matching
        input_words = input_text.split()
        if len(input_words) > 8:
            return 0.0
        
        # Fast method: Check if most words from input appear in reference
        ref_words_set = set(reference_text.split())
        matching_words = sum(1 for word in input_words if word in ref_words_set)
        word_coverage = matching_words / len(input_words) if input_words else 0
        
        # If most words match, do more detailed similarity check on smaller segments
        if word_coverage >= 0.6:
            # Find the best matching subsequence using a more efficient approach
            ref_words = reference_text.split()
            input_len = len(input_words)
            
            # Limit search to avoid O(n²) complexity
            max_positions = min(20, len(ref_words) - input_len + 1)  # Reduced from 50
            step_size = max(1, (len(ref_words) - input_len) // max_positions) if max_positions > 0 else 1
            
            best_similarity = 0.0
            positions_checked = 0
            for i in range(0, len(ref_words) - input_len + 1, step_size):
                window = ' '.join(ref_words[i:i + input_len])
                similarity = SequenceMatcher(None, input_text, window).ratio()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    
                # Early exit for very good matches
                if similarity >= 0.9:
                    best_similarity = min(1.0, similarity + 0.05)
                    break
                    
                positions_checked += 1
                if positions_checked >= max_positions:  # Hard limit to prevent runaway
                    break
            
            # Boost good fuzzy matches but be more selective
            if best_similarity >= 0.8:
                return min(1.0, best_similarity + 0.1)
            elif best_similarity >= 0.7:
                return best_similarity
        
        return 0.0
    
    def _get_adaptive_threshold(self, base_threshold: float, word_count: int, match_type: str) -> float:
        """
        Get adaptive threshold based on input characteristics and match type.
        
        Args:
            base_threshold (float): Original threshold
            word_count (int): Number of words in input
            match_type (str): Type of match found
            
        Returns:
            float: Adaptive threshold
        """
        # For short inputs (likely hadith excerpts), use lower thresholds
        if word_count <= 3:
            if match_type == 'substring':
                return 0.75
            elif match_type == 'fuzzy_substring':
                return 0.80
            else:
                return max(0.65, base_threshold - 0.2)
        elif word_count <= 8:
            if match_type == 'substring':
                return 0.75
            elif match_type == 'fuzzy_substring':
                return 0.85
            else:
                return max(0.75, base_threshold - 0.1)
        else:
            # For longer inputs, use closer to original threshold
            return max(0.85, base_threshold - 0.05)
            
        match_ratio = results['matched_count'] / results['total_count']
        
        if self.multiverse_strategy == 'any':
            return results['matched_count'] > 0
        elif self.multiverse_strategy == 'majority':
            return match_ratio > 0.5
        elif self.multiverse_strategy == 'all':
            return match_ratio == 1.0
        elif self.multiverse_strategy == 'high':
            return match_ratio >= 0.8
        else:
            # Default to 'any' strategy
            return results['matched_count'] > 0


