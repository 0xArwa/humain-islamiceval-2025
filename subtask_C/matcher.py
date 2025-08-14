import json
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
import re
from difflib import SequenceMatcher
from datetime import datetime
import time
import zipfile
import traceback
import math
from typing import Dict, List, Set, Optional, Tuple
from reranker import ReRanker


class QuranHadithSpanMatcher:
    """
    Matcher with cross-ayah support and separate thresholds for Quran and Hadith
    includes extracted spans storage in JSON format and re-ranking
    """
    
    def __init__(self, quran_index_dir="quran_index", hadith_index_dir="hadith_index", 
                 verbose=False, use_hf_reranker=True, reranker_model="BAAI/bge-reranker-v2-m3"):
        self.quran_index_dir = Path(quran_index_dir)
        self.hadith_index_dir = Path(hadith_index_dir)
        self.verbose = verbose
        
        # Initialize re-ranker
        self.use_hf_reranker = use_hf_reranker
        self.reranker = None
        if use_hf_reranker:
            try:
                self.reranker = ReRanker(
                    model_name=reranker_model,
                    verbose=verbose
                )
            except Exception as e:
                if verbose:
                    print(f"Failed to initialize HuggingFace re-ranker: {e}")
                    print("Will use rule-based re-ranking instead")
                self.use_hf_reranker = False
                self.reranker = None
        
        # Indexes
        self.quran_word_index = {}
        self.quran_ayah_lookup = {}
        self.hadith_word_index = {}
        self.hadith_lookup = {}
        
        # Caches for performance
        self.normalization_cache = {}
        self.quran_normalized_texts = {}
        self.hadith_normalized_texts = {}
        
        # Inverted indices for fast substring search
        self.quran_ngram_index = defaultdict(set)
        self.hadith_ngram_index = defaultdict(set)
        
        # SEPARATE THRESHOLDS for Quran and Hadith
        self.quran_similarity_threshold = 0.85  # Higher threshold for Quran
        self.hadith_similarity_threshold = 0.75  # Lower threshold for Hadith
        self.min_word_matches = 1
        self.ngram_size = 3
        
        # Verse separator patterns - ADDED * as separator
        self.verse_separator_patterns = [
            r'\*',  # Added asterisk as separator
            r'\([٠-٩]+\)', r'\([0-9]+\)', r'﴿[٠-٩]+﴾',
            r'﴿[0-9]+﴾', r'\[[٠-٩]+\]', r'\[[0-9]+\]'
        ]
        
        # Compile regex patterns once
        self.arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.arabic_english_numbers = re.compile(r'[\u06F0-\u06F9\u0660-\u0669\d]')
        
        # Load and optimize indexes
        self.load_indexes()
        self.build_indices()
    
    def get_threshold_for_source(self, source_type: str) -> float:
        """Get appropriate threshold based on source type"""
        if source_type == 'quran':
            return self.quran_similarity_threshold
        elif source_type == 'hadith':
            return self.hadith_similarity_threshold
        else:
            # Fallback to hadith threshold for unknown types
            return self.hadith_similarity_threshold
    
    def normalize_text(self, text: str) -> str:
        """Normalize Arabic text with caching - PROPERLY FIXED VERSION"""
        if not text:
            return ""
        
        if text in self.normalization_cache:
            return self.normalization_cache[text]
        
        # Remove diacritics first
        normalized = self.arabic_diacritics.sub('', text)
        
        # Remove punctuation but REPLACE with spaces (don't just remove)
        punctuation_set = {'ۚ', '۩', '﴾', '﴿', '؛', '،', '.', ',', '!', '?', ':', ';', '"', "'", '(', ')', '[', ']', '{', '}'}
        # Replace each punctuation mark with a space
        for punct in punctuation_set:
            normalized = normalized.replace(punct, ' ')
        
        # Remove numbers
        normalized = self.arabic_english_numbers.sub('', normalized)
        
        # CRITICAL: Properly normalize whitespace to split words correctly
        normalized = ' '.join(normalized.split())
        
        # Store and return the NORMALIZED text (not original)
        self.normalization_cache[text] = normalized
        return normalized
    
    def generate_ngrams(self, text: str, n: int = 3) -> Set[str]:
        """Generate n-grams from text for indexing"""
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def build_indices(self):
        """Build optimized indices for fast searching"""
        if self.verbose:
            print("Building optimized indices...")
            start_time = time.time()
        
        # Build n-gram index for Quran
        for doc_id, doc_data in self.quran_ayah_lookup.items():
            if 'normalized_text' in doc_data:
                normalized = doc_data['normalized_text']
                ngrams = self.generate_ngrams(normalized, self.ngram_size)
                for ngram in ngrams:
                    self.quran_ngram_index[ngram].add(doc_id)
        
        # Build n-gram index for Hadith
        for doc_id, doc_data in self.hadith_lookup.items():
            if 'Matn' in doc_data and doc_data['Matn']:
                matn_normalized = self.normalize_text(doc_data['Matn'])
                self.hadith_normalized_texts[f"{doc_id}_matn"] = matn_normalized
                ngrams = self.generate_ngrams(matn_normalized, self.ngram_size)
                for ngram in ngrams:
                    self.hadith_ngram_index[ngram].add(f"{doc_id}_matn")
            
            if 'hadithTxt' in doc_data and doc_data['hadithTxt']:
                hadith_normalized = self.normalize_text(doc_data['hadithTxt'])
                self.hadith_normalized_texts[f"{doc_id}_hadith"] = hadith_normalized
                ngrams = self.generate_ngrams(hadith_normalized, self.ngram_size)
                for ngram in ngrams:
                    self.hadith_ngram_index[ngram].add(f"{doc_id}_hadith")
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"  ✓ Indices built in {elapsed:.2f} seconds")
    
    def load_indexes(self):
        """Load indexes with progress tracking"""
        if self.verbose:
            print("Loading Quran and Hadith indexes...")
        
        # Load Quran index
        try:
            quran_word_file = self.quran_index_dir / "quran_word_index.json"
            with open(quran_word_file, 'r', encoding='utf-8') as f:
                word_index_data = json.load(f)
                self.quran_word_index = {
                    word: set(ayah_ids) for word, ayah_ids in word_index_data.items()
                }
            
            quran_lookup_file = self.quran_index_dir / "quran_ayah_lookup.json"
            with open(quran_lookup_file, 'r', encoding='utf-8') as f:
                self.quran_ayah_lookup = json.load(f)
            
            # Pre-normalize all Quran texts
            for ayah_id, ayah_data in self.quran_ayah_lookup.items():
                if 'ayah_text' in ayah_data:
                    ayah_data['normalized_text'] = self.normalize_text(ayah_data['ayah_text'])
            
            if self.verbose:
                print(f"  ✓ Quran index: {len(self.quran_word_index):,} words, {len(self.quran_ayah_lookup):,} ayahs")
        
        except FileNotFoundError as e:
            print(f"Warning: Could not load Quran index: {e}")
        
        # Load Hadith index
        try:
            hadith_word_file = self.hadith_index_dir / "hadith_word_index.json"
            with open(hadith_word_file, 'r', encoding='utf-8') as f:
                word_index_data = json.load(f)
                self.hadith_word_index = {
                    word: set(hadith_ids) for word, hadith_ids in word_index_data.items()
                }
            
            hadith_lookup_file = self.hadith_index_dir / "hadith_lookup.json"
            with open(hadith_lookup_file, 'r', encoding='utf-8') as f:
                self.hadith_lookup = json.load(f)
            
            if self.verbose:
                print(f"  ✓ Hadith index: {len(self.hadith_word_index):,} words, {len(self.hadith_lookup):,} hadiths")
        
        except FileNotFoundError as e:
            print(f"Warning: Could not load Hadith index: {e}")
    
    def generate_word_ngrams(self, text: str, n: int) -> List[str]:
        """Generate word-based n-grams from text"""
        if not text or n <= 0:
            return []
        
        words = text.split()
        if len(words) < n:
            return [' '.join(words)]
        
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def calculate_ngram_overlap_score(self, query: str, candidate: str, ngram_sizes: List[int] = [2, 3, 4]) -> float:
        """Calculate weighted n-gram overlap score"""
        query_norm = self.normalize_text(query)
        candidate_norm = self.normalize_text(candidate)
        
        if not query_norm or not candidate_norm:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Weight different n-gram sizes (larger n-grams get higher weight)
        weights = {2: 0.3, 3: 0.4, 4: 0.3}
        
        for n in ngram_sizes:
            if n in weights:
                query_ngrams = set(self.generate_word_ngrams(query_norm, n))
                candidate_ngrams = set(self.generate_word_ngrams(candidate_norm, n))
                
                if query_ngrams and candidate_ngrams:
                    overlap = len(query_ngrams & candidate_ngrams)
                    total_ngrams = len(query_ngrams | candidate_ngrams)
                    
                    if total_ngrams > 0:
                        ngram_score = overlap / total_ngrams
                        total_score += ngram_score * weights[n]
                        total_weight += weights[n]
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    
    def calculate_phrase_matching_score(self, query: str, candidate: str) -> float:
        """Calculate score based on consecutive phrase matching"""
        query_norm = self.normalize_text(query)
        candidate_norm = self.normalize_text(candidate)
        
        if not query_norm or not candidate_norm:
            return 0.0
        
        query_words = query_norm.split()
        candidate_words = candidate_norm.split()
        
        if not query_words:
            return 0.0
        
        max_consecutive = 0
        total_consecutive = 0
        
        # Find all consecutive matches
        for i in range(len(query_words)):
            for j in range(len(candidate_words)):
                consecutive_count = 0
                
                # Count consecutive matching words
                qi, cj = i, j
                while (qi < len(query_words) and cj < len(candidate_words) and 
                       query_words[qi] == candidate_words[cj]):
                    consecutive_count += 1
                    qi += 1
                    cj += 1
                
                if consecutive_count > 0:
                    max_consecutive = max(max_consecutive, consecutive_count)
                    total_consecutive += consecutive_count
        
        # Normalize by query length
        max_consecutive_score = max_consecutive / len(query_words)
        coverage_score = min(total_consecutive / len(query_words), 1.0)
        
        # Weighted combination
        return max_consecutive_score * 0.6 + coverage_score * 0.4
    
    def calculate_substring_containment_score(self, query: str, candidate: str) -> float:
        """Calculate how well the candidate contains substrings from the query"""
        query_norm = self.normalize_text(query)
        candidate_norm = self.normalize_text(candidate)
        
        if not query_norm or not candidate_norm:
            return 0.0
        
        # Calculate character-level containment
        query_chars = set(query_norm.replace(' ', ''))
        candidate_chars = set(candidate_norm.replace(' ', ''))
        
        if not query_chars:
            return 0.0
        
        char_overlap = len(query_chars & candidate_chars) / len(query_chars)
        
        # Calculate word-level containment
        query_words = query_norm.split()
        candidate_words = candidate_norm.split()
        
        if not query_words:
            return char_overlap
        
        word_overlap = len(set(query_words) & set(candidate_words)) / len(set(query_words))
        
        # Check for substring containment (both directions)
        substring_score = 0.0
        if query_norm in candidate_norm:
            substring_score = 1.0
        elif candidate_norm in query_norm:
            substring_score = 0.8
        else:
            # Check for partial substring matches
            max_substr_len = 0
            for i in range(len(query_norm)):
                for j in range(i + 3, len(query_norm) + 1):  # minimum 3 characters
                    substr = query_norm[i:j]
                    if substr in candidate_norm:
                        max_substr_len = max(max_substr_len, len(substr))
            
            if max_substr_len > 0:
                substring_score = max_substr_len / len(query_norm)
        
        # Combine scores
        return char_overlap * 0.3 + word_overlap * 0.4 + substring_score * 0.3
    
    def calculate_positional_score(self, query: str, candidate: str) -> float:
        """Calculate score based on word position consistency"""
        query_norm = self.normalize_text(query)
        candidate_norm = self.normalize_text(candidate)
        
        if not query_norm or not candidate_norm:
            return 0.0
        
        query_words = query_norm.split()
        candidate_words = candidate_norm.split()
        
        if not query_words or not candidate_words:
            return 0.0
        
        # Find positions of query words in candidate
        matched_positions = []
        for i, qword in enumerate(query_words):
            for j, cword in enumerate(candidate_words):
                if qword == cword:
                    matched_positions.append((i, j))
                    break  # Take first match
        
        if not matched_positions:
            return 0.0
        
        # Calculate position consistency score
        position_score = len(matched_positions) / len(query_words)
        
        # Bonus for maintaining relative order
        if len(matched_positions) >= 2:
            order_preserved = 0
            for k in range(len(matched_positions) - 1):
                if matched_positions[k][1] < matched_positions[k+1][1]:
                    order_preserved += 1
            
            order_score = order_preserved / (len(matched_positions) - 1)
            position_score = position_score * 0.7 + order_score * 0.3
        
        return position_score
    
    def fallback_matching(self, query: str, source_type: str, min_ngram_score: float = 0.3) -> List[Dict]:
        """Fallback matching using n-grams, important words, and phrase matching"""
        if source_type == 'quran':
            lookup_dict = self.quran_word_index
            data_dict = self.quran_ayah_lookup
        elif source_type == 'hadith':
            lookup_dict = self.hadith_word_index
            data_dict = self.hadith_lookup
        else:
            return []
        
        if not query or not lookup_dict:
            return []
        
        normalized_query = self.normalize_text(query)
        query_words = set(normalized_query.split())
        
        if not query_words:
            return []
        
        # Find candidates with at least some word overlap
        candidates = []
        word_counts = Counter()
        
        for word in query_words:
            if word in lookup_dict:
                for doc_id in lookup_dict[word]:
                    word_counts[doc_id] += 1
        
        # Require at least 1 word match for fallback
        candidates = [doc_id for doc_id, count in word_counts.items() if count >= 1]
        
        if not candidates:
            return []
        
        # Limit candidates for performance - reduced for speed
        if len(candidates) > 100:
            # Priority to candidates with more word overlap
            candidates = [doc_id for doc_id, _ in word_counts.most_common(100)]
        
        matches = []
        
        for doc_id in candidates:
            if source_type == 'quran':
                doc_text = data_dict.get(doc_id, {}).get('text', '')
            else:  # hadith
                doc_text = data_dict.get(doc_id, {}).get('Matn', '')
            
            if not doc_text:
                continue
            
            # Calculate multiple similarity metrics
            ngram_score = self.calculate_ngram_overlap_score(query, doc_text)
            phrase_score = self.calculate_phrase_matching_score(query, doc_text)
            
            # NEW: Add substring containment check
            substring_score = self.calculate_substring_containment_score(query, doc_text)
            
            # Calculate composite score with substring consideration
            composite_score = (
                ngram_score * 0.4 +
                phrase_score * 0.35 +
                substring_score * 0.25
            )
            
            # Apply minimum thresholds - more lenient for fallback
            if (composite_score >= min_ngram_score or 
                ngram_score >= 0.2 or  # More lenient n-gram threshold
                phrase_score >= 0.25 or  # More lenient phrase matching
                substring_score >= 0.4):  # More lenient for substring matches
                
                # Create match object similar to existing structure
                if source_type == 'quran':
                    match = data_dict[doc_id].copy()
                    match['type'] = 'quran'
                else:
                    match = data_dict[doc_id].copy()
                    match['type'] = 'hadith'
                
                match['similarity'] = composite_score
                match['match_type'] = 'fallback_enhanced'
                match['ngram_score'] = ngram_score
                match['phrase_score'] = phrase_score
                match['substring_score'] = substring_score
                
                matches.append(match)
        
        # Sort by composite score and return top matches
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:20]  # Return top 20 fallback matches
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Re-rank candidate matches using HuggingFace model or fallback to rule-based"""
        if not candidates:
            return candidates
        
        if len(candidates) <= 1:
            return candidates
        
        # Use HuggingFace re-ranker if available
        if self.use_hf_reranker and self.reranker and self.reranker.is_available():
            if self.verbose:
                print(f"Using HuggingFace re-ranker for {len(candidates)} candidates")
            
            rerank_result = self.reranker.rerank_candidates(
                query=query,
                candidates=candidates,
                top_k=top_k,
                combine_with_original=True,
                alpha=0.6  # Weight for original scores
            )
            
            return rerank_result.candidates
        
        else:
            # Fallback to rule-based re-ranking
            if self.verbose:
                print(f"Using rule-based re-ranking for {len(candidates)} candidates")
            
            if self.reranker:
                return self.reranker.fallback_rerank_candidates(query, candidates, top_k)
            else:
                # Simple fallback if no re-ranker at all
                return candidates[:top_k]
    
    def calculate_sequence_similarity(self, text1: str, text2: str) -> float:
        """Calculate sequence similarity using SequenceMatcher"""
        if not text1 or not text2:
            return 0.0
        
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        if not norm1 or not norm2:
            return 0.0
        
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def find_exact_substring_matches(self, normalized_query: str, source_type: str) -> List[Dict]:
        """Fast exact substring matching using n-gram index"""
        matches = []
        
        if source_type == 'quran':
            query_ngrams = self.generate_ngrams(normalized_query, self.ngram_size)
            if not query_ngrams:
                return matches
            
            candidate_docs = None
            for ngram in query_ngrams:
                if ngram in self.quran_ngram_index:
                    if candidate_docs is None:
                        candidate_docs = self.quran_ngram_index[ngram].copy()
                    else:
                        candidate_docs &= self.quran_ngram_index[ngram]
            
            if not candidate_docs:
                return matches
            
            for doc_id in candidate_docs:
                doc_data = self.quran_ayah_lookup[doc_id]
                if 'normalized_text' in doc_data:
                    if normalized_query in doc_data['normalized_text']:
                        matches.append({
                            'type': 'quran',
                            'id': doc_id,
                            'surah_id': doc_data['surah_id'],
                            'ayah_id': doc_data['ayah_id'],
                            'surah_name': doc_data['surah_name'],
                            'text': doc_data['ayah_text'],
                            'similarity': 1.0,
                            'match_type': 'exact_normalized'
                        })
                        return matches
        
        else:  # hadith
            query_ngrams = self.generate_ngrams(normalized_query, self.ngram_size)
            if not query_ngrams:
                return matches
            
            candidate_docs = None
            for ngram in query_ngrams:
                if ngram in self.hadith_ngram_index:
                    if candidate_docs is None:
                        candidate_docs = self.hadith_ngram_index[ngram].copy()
                    else:
                        candidate_docs &= self.hadith_ngram_index[ngram]
            
            if not candidate_docs:
                return matches
            
            for doc_key in candidate_docs:
                if doc_key in self.hadith_normalized_texts:
                    if normalized_query in self.hadith_normalized_texts[doc_key]:
                        doc_id = doc_key.rsplit('_', 1)[0]
                        field = doc_key.rsplit('_', 1)[1]
                        
                        doc_data = self.hadith_lookup[doc_id]
                        text = doc_data.get('Matn', '') if field == 'matn' else doc_data.get('hadithTxt', '')
                        
                        matches.append({
                            'type': 'hadith',
                            'id': doc_id,
                            'hadithID': doc_data['hadithID'],
                            'BookID': doc_data['BookID'],
                            'title': doc_data['title'],
                            'text': text,
                            'similarity': 1.0,
                            'match_type': f'exact_normalized_{field}'
                        })
                        return matches
        
        return matches
    
    def find_cross_ayah_candidates(self, query_words: List[str]) -> List[str]:
        """Find ayahs that could be starting points for cross-ayah matches"""
        word_matches = {}
        
        # Count word matches per ayah - use more words for Arabic
        search_words = query_words[:min(8, len(query_words))]  # Up to 8 words
        
        for word in search_words:
            # Skip very short words
            if len(word) < 2:
                continue
                
            if word in self.quran_word_index:
                for ayah_id in self.quran_word_index[word]:
                    word_matches[ayah_id] = word_matches.get(ayah_id, 0) + 1
        
        # Sort by word match count and return top candidates
        sorted_candidates = sorted(word_matches.items(), key=lambda x: x[1], reverse=True)
        
        # More lenient minimum match requirement
        min_matches = max(1, len(search_words) // 4)  # At least 25% of words should match
        
        candidates = [ayah_id for ayah_id, count in sorted_candidates if count >= min_matches]
        
        if self.verbose:
            print(f"Word matching: {len(search_words)} search words, {min_matches} min matches")
            print(f"Found {len(candidates)} candidates from word index")
        
        return candidates
    
    def combine_consecutive_ayahs(self, surah_num: int, start_ayah: int, num_ayahs: int, normalized_query: str) -> Dict:
        """Combine consecutive ayahs and check similarity"""
        
        combined_texts = []
        ayah_details = []
        
        # Collect consecutive ayahs
        for i in range(num_ayahs):
            ayah_id = f"ayah_{surah_num}_{start_ayah + i}"
            if ayah_id not in self.quran_ayah_lookup:
                return None  # Gap in ayahs, can't combine
            
            ayah_data = self.quran_ayah_lookup[ayah_id]
            if 'normalized_text' not in ayah_data:
                return None
            
            combined_texts.append(ayah_data['normalized_text'])
            ayah_details.append({
                'ayah_id': start_ayah + i,
                'text': ayah_data['ayah_text']
            })
        
        # Combine all texts
        combined_normalized = ' '.join(combined_texts)
        
        # Calculate similarity
        similarity = self.calculate_sequence_similarity(normalized_query, combined_normalized)
        
        if similarity >= self.quran_similarity_threshold:
            # Format combined text with ayah numbers
            formatted_parts = []
            for detail in ayah_details:
                formatted_parts.append(f"{detail['text']}({detail['ayah_id']})")
            
            return {
                'type': 'quran',
                'similarity': similarity,
                'match_type': 'cross_ayah',
                'surah_id': surah_num,
                'surah_name': self.quran_ayah_lookup[f"ayah_{surah_num}_{start_ayah}"].get('surah_name', ''),
                'ayah_id': start_ayah,  # First ayah
                'ayah_ids': [detail['ayah_id'] for detail in ayah_details],
                'text': ' '.join(formatted_parts),
                'formatted_text': ' '.join(formatted_parts),
                'num_ayahs_combined': num_ayahs,
                'is_cross_ayah': True,
                'threshold_used': self.quran_similarity_threshold
            }
        
        return None
    
    def find_cross_ayah_matches(self, query_text: str) -> List[Dict]:
        """Find matches that span multiple consecutive ayahs"""
        normalized_query = self.normalize_text(query_text)
        query_words = normalized_query.split()
        
        if len(query_words) < 4:  # Too short for cross-ayah
            return []
        
        candidates = self.find_cross_ayah_candidates(query_words)[:30]  # Limit for performance - optimized
        
        if not candidates:
            return []
        
        matches = []
        
        for candidate_id in candidates:
            # Parse ayah ID to get surah and ayah numbers
            try:
                parts = candidate_id.split('_')
                if len(parts) >= 3:
                    surah_num = int(parts[1])
                    start_ayah = int(parts[2])
                else:
                    continue
            except (ValueError, IndexError):
                continue
            
            # Try combining 2-5 consecutive ayahs starting from this one
            for num_ayahs in range(2, 6):
                combined_match = self.combine_consecutive_ayahs(surah_num, start_ayah, num_ayahs, normalized_query)
                if combined_match:
                    matches.append(combined_match)
        
        # Sort by similarity and number of ayahs (prefer fewer ayahs for same similarity)
        matches.sort(key=lambda x: (x['similarity'], -x['num_ayahs_combined']), reverse=True)
        
        if self.verbose and matches:
            top_match = matches[0]
            print(f"Cross-ayah: found {len(matches)} candidates, "
                  f"top similarity={top_match['similarity']:.3f}, "
                  f"ayahs={top_match['num_ayahs_combined']}")
        
        return matches[:5]  # Return top 5 cross-ayah matches
    
    def find_single_ayah_matches(self, query_text: str, source_type: str) -> List[Dict]:
        """Find matches within single ayahs or hadiths"""
        if not query_text:
            return []
        
        normalized_query = self.normalize_text(query_text)
        
        # First try exact substring matching (fastest)
        exact_matches = self.find_exact_substring_matches(normalized_query, source_type)
        if exact_matches:
            return exact_matches
        
        matches = []
        
        if source_type == 'quran':
            # Get word-based candidates
            query_words = set(normalized_query.split())
            candidates = []
            word_counts = Counter()
            
            for word in query_words:
                if word in self.quran_word_index:
                    for ayah_id in self.quran_word_index[word]:
                        word_counts[ayah_id] += 1
            
            # Filter candidates by minimum word overlap
            min_overlap = max(1, len(query_words) // 3)
            candidates = [ayah_id for ayah_id, count in word_counts.items() if count >= min_overlap]
            
            # Limit for performance - optimized
            if len(candidates) > 100:
                candidates = [ayah_id for ayah_id, _ in word_counts.most_common(100)]
            
            # Calculate similarities
            for ayah_id in candidates:
                ayah_data = self.quran_ayah_lookup.get(ayah_id)
                if not ayah_data or 'normalized_text' not in ayah_data:
                    continue
                
                similarity = self.calculate_sequence_similarity(normalized_query, ayah_data['normalized_text'])
                
                # Early exit for very high similarity
                if similarity >= 0.95:
                    return [{
                        'type': 'quran',
                        'id': ayah_id,
                        'surah_id': ayah_data['surah_id'],
                        'ayah_id': ayah_data['ayah_id'],
                        'surah_name': ayah_data['surah_name'],
                        'text': ayah_data['ayah_text'],
                        'similarity': similarity,
                        'match_type': 'high_confidence_early_exit',
                        'threshold_used': self.quran_similarity_threshold
                    }]
                
                if similarity >= 0.15:  # Lower threshold for initial filtering
                    matches.append({
                        'type': 'quran',
                        'id': ayah_id,
                        'surah_id': ayah_data['surah_id'],
                        'ayah_id': ayah_data['ayah_id'],
                        'surah_name': ayah_data['surah_name'],
                        'text': ayah_data['ayah_text'],
                        'similarity': similarity,
                        'match_type': 'single_ayah',
                        'threshold_used': self.quran_similarity_threshold
                    })
        
        else:  # hadith
            query_words = set(normalized_query.split())
            candidates = []
            word_counts = Counter()
            
            for word in query_words:
                if word in self.hadith_word_index:
                    for hadith_id in self.hadith_word_index[word]:
                        word_counts[hadith_id] += 1
            
            min_overlap = max(1, len(query_words) // 3)
            candidates = [hadith_id for hadith_id, count in word_counts.items() if count >= min_overlap]
            
            if len(candidates) > 100:
                candidates = [hadith_id for hadith_id, _ in word_counts.most_common(100)]
            
            for hadith_id in candidates:
                hadith_data = self.hadith_lookup.get(hadith_id)
                if not hadith_data:
                    continue
                
                # Check both Matn and hadithTxt fields
                for field, text in [('Matn', hadith_data.get('Matn', '')), ('hadithTxt', hadith_data.get('hadithTxt', ''))]:
                    if not text:
                        continue
                    
                    normalized_text = self.normalize_text(text)
                    similarity = self.calculate_sequence_similarity(normalized_query, normalized_text)
                    
                    # Early exit for very high similarity in hadith
                    if similarity >= 0.95:
                        return [{
                            'type': 'hadith',
                            'id': hadith_id,
                            'hadithID': hadith_data['hadithID'],
                            'BookID': hadith_data['BookID'],
                            'title': hadith_data['title'],
                            'text': text,
                            'similarity': similarity,
                            'match_type': f'high_confidence_early_exit_{field}',
                            'threshold_used': self.hadith_similarity_threshold
                        }]
                    
                    if similarity >= 0.15:
                        matches.append({
                            'type': 'hadith',
                            'id': hadith_id,
                            'hadithID': hadith_data['hadithID'],
                            'BookID': hadith_data['BookID'],
                            'title': hadith_data['title'],
                            'text': text,
                            'similarity': similarity,
                            'match_type': f'single_hadith_{field}',
                            'threshold_used': self.hadith_similarity_threshold
                        })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:20]  # Return top 20 matches
    
    def find_matches_by_sequence(self, query_text: str, source_type: str) -> List[Dict]:
        """Main matching function that combines all strategies"""
        all_matches = []
        
        # Strategy 1: single ayah/hadith matches
        single_matches = self.find_single_ayah_matches(query_text, source_type)
        all_matches.extend(single_matches)
        
        # Strategy 2: cross-ayah matches (only for Quran)
        if source_type == 'quran':
            cross_matches = self.find_cross_ayah_matches(query_text)
            all_matches.extend(cross_matches)
        
        # Strategy 3: fallback matching if no good matches found
        threshold = self.get_threshold_for_source(source_type)
        if not all_matches or all_matches[0]['similarity'] < threshold:
            fallback_matches = self.fallback_matching(query_text, source_type)
            all_matches.extend(fallback_matches)
        
        # Remove duplicates (keep highest similarity)
        seen_ids = {}
        unique_matches = []
        for match in all_matches:
            match_id = match.get('id', f"{match.get('type')}_{match.get('surah_id', '')}_{match.get('ayah_id', '')}_{match.get('hadithID', '')}")
            if match_id not in seen_ids or match['similarity'] > seen_ids[match_id]['similarity']:
                seen_ids[match_id] = match
        
        unique_matches = list(seen_ids.values())
        unique_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Skip reranking if we have a very high confidence match
        if unique_matches and unique_matches[0]['similarity'] >= 0.95:
            if self.verbose:
                print(f"Skipping reranking due to high confidence: {unique_matches[0]['similarity']:.3f}")
            return unique_matches[:10]  # Return top 10 without reranking
        
        # Apply re-ranking to top candidates
        if len(unique_matches) > 1:
            reranked_matches = self.rerank_candidates(query_text, unique_matches, top_k=10)
            return reranked_matches
        
        return unique_matches
    
    def split_verses_if_needed(self, text: str) -> List[str]:
        """Split text into individual verses if separators are found"""
        if not text or not text.strip():
            return [text]
        
        separators_found = []
        for pattern in self.verse_separator_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                separators_found.extend([(match.start(), match.end(), match.group()) for match in matches])
        
        if not separators_found:
            return [text.strip()]
        
        separators_found.sort(key=lambda x: x[0])
        
        verses = []
        last_end = 0
        
        for start_pos, end_pos, separator in separators_found:
            if start_pos > last_end:
                verse_text = text[last_end:start_pos].strip()
                if verse_text:
                    verses.append(verse_text)
            last_end = end_pos
        
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                verses.append(remaining_text)
        
        return verses if verses else [text.strip()]
    
    def match_span_with_verse_splitting(self, original_span: str, source_type: str, verbose: bool = False) -> str:
        """Match a text span with verse splitting capability and source-specific thresholds"""
        if not original_span or original_span.strip() == "خطأ":
            return "خطأ"
        
        query_text = original_span.strip()
        verses = self.split_verses_if_needed(query_text)
        
        if len(verses) == 1:
            return self.match_single_verse(verses[0], source_type)
        else:
            return self.match_multiple_verses(verses, source_type)
    
    def match_single_verse(self, verse_text: str, source_type: str):
        """Match a single verse with source-specific thresholds"""
        if not verse_text:
            return "خطأ"
        
        if self.verbose:
            print(f"Matching single verse: '{verse_text[:50]}...' (source: {source_type})")
        
        # Use source-specific matching
        matches = self.find_matches_by_sequence(verse_text, source_type)
        threshold = self.get_threshold_for_source(source_type)
        if matches and matches[0]['similarity'] >= threshold:
            if self.verbose:
                match_type = matches[0].get('match_type', 'unknown')
                is_cross = 'cross_ayah' in match_type
                print(f"Found match: {match_type} ({'cross-ayah' if is_cross else 'single-ayah'})")
            return self.format_match_result(matches[0])
        
        # HF Reranker fallback
        if (matches and matches[0].get('hf_rerank_score', 0) > 0.5):
            if self.verbose:
                print(f"Using HF reranker fallback: HF score = {matches[0]['hf_rerank_score']:.3f}")
            matches[0]['match_type'] = matches[0].get('match_type', '') + '_hf_fallback'
            return self.format_match_result(matches[0])
        
        if self.verbose:
            print("No match found")
        
        return "خطأ"
    
    def format_match_result(self, match: Dict):
        """Format match result into standard structure with re-ranking info"""
        if match['type'] == 'quran':
            result = {
                'source': 'quran',
                'surah_id': match['surah_id'],
                'ayah_id': match['ayah_id'],
                'surah_name': match['surah_name'],
                'confidence': match['similarity'],
                'match_type': match.get('match_type', 'similarity'),
                'threshold_used': match.get('threshold_used', self.quran_similarity_threshold)
            }
            
            # Add re-ranking information if available
            if 'rerank_score' in match:
                result['rerank_score'] = match['rerank_score']
                result['original_rank'] = match.get('original_rank', 1)
                result['rerank_features'] = match.get('rerank_features', {})
            
            # Add cross-ayah specific fields if present
            if 'cross_ayah' in match.get('match_type', ''):
                result['ayah_ids'] = match.get('ayah_ids', [])
                result['num_ayahs_combined'] = match.get('num_ayahs_combined', 1)
                result['is_cross_ayah'] = True
                
                # Use formatted text with ayah numbers (like multiple verses)
                result['text'] = match.get('formatted_text', match['text'])
            else:
                result['text'] = match['text']
                result['is_cross_ayah'] = False
            
            return result
        else:
            result = {
                'source': 'hadith',
                'hadithID': match['hadithID'],
                'BookID': match['BookID'],
                'title': match['title'],
                'text': match.get('text', match.get('Matn', '')),  # Handle both 'text' and 'Matn' fields
                'confidence': match['similarity'],
                'match_type': match.get('match_type', 'similarity'),
                'threshold_used': match.get('threshold_used', self.hadith_similarity_threshold),
                'is_cross_ayah': False
            }
            
            # Add re-ranking information if available
            if 'rerank_score' in match:
                result['rerank_score'] = match['rerank_score']
                result['original_rank'] = match.get('original_rank', 1)
                result['rerank_features'] = match.get('rerank_features', {})
            
            return result
    
    def match_multiple_verses(self, verses: List[str], source_type: str):
        """Process multiple verses with source-specific thresholds"""
        all_matches = []
        
        for i, verse in enumerate(verses):
            # Try to find matches for each verse
            matches = self.find_matches_by_sequence(verse, source_type)
            threshold = self.get_threshold_for_source(source_type)
            if matches and matches[0]['similarity'] >= threshold:
                result = self.format_match_result(matches[0])
                result['verse_index'] = i
                all_matches.append(result)
        
        if not all_matches:
            return "خطأ"
        
        # Build the corrected text with ayah numbers
        if len(all_matches) == 1:
            # Single match - return just the text without number
            return all_matches[0]
        else:
            # Multiple matches - combine with ayah numbers
            corrected_parts = []
            for match in all_matches:
                if match['source'] == 'quran':
                    # Add the ayah with its number in parentheses
                    corrected_parts.append(f"{match['text']}({match['ayah_id']})")
                else:
                    # For hadith, just add the text
                    corrected_parts.append(match['text'])
            
            # Create a combined result with all matched verses
            combined_result = {
                'source': all_matches[0]['source'],  # Use first match source
                'text': ' '.join(corrected_parts),  # Combined text with ayah numbers
                'confidence': min(m['confidence'] for m in all_matches),  # Minimum confidence
                'match_type': 'multiple_verses',
                'all_matches': all_matches,  # Store all individual matches
                'original_verses': verses,
                'matched_verse_count': len(all_matches),
                'total_verse_count': len(verses),
                'threshold_used': all_matches[0].get('threshold_used', 0.0)
            }
            
            # Add metadata from first match
            if all_matches[0]['source'] == 'quran':
                combined_result['surah_id'] = all_matches[0]['surah_id']
                combined_result['surah_name'] = all_matches[0]['surah_name']
                # Store all ayah IDs
                combined_result['ayah_ids'] = [m['ayah_id'] for m in all_matches]
            else:
                combined_result['hadithID'] = all_matches[0]['hadithID']
                combined_result['BookID'] = all_matches[0]['BookID']
                combined_result['title'] = all_matches[0]['title']
            
            return combined_result
