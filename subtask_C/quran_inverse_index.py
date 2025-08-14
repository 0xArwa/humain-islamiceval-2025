import json
import re
from collections import defaultdict
from pathlib import Path

class QuranInverseIndexBuilder:
    """
    Builds an optimized inverse index for Quran text matching.
    Works with both exact ayah text and normalized version (without tashkeel).
    """
    
    def __init__(self, min_word_length=1, save_index=True):
        self.min_word_length = min_word_length
        self.save_index = save_index
        
        # Unified inverse index: word -> set of ayah_ids
        # Contains both exact words (with tashkeel) and normalized words (without tashkeel)
        self.word_index = defaultdict(set)
        
        # Ayah lookup: ayah_id -> ayah_data
        self.ayah_lookup = {}
        
        # Statistics
        self.stats = {
            'total_ayahs': 0,
            'unique_words': 0,
            'total_word_entries': 0,
            'exact_words_added': 0,
            'normalized_words_added': 0
        }
    
    def normalize_arabic_text(self, text):
        """
        Normalize Arabic text by removing diacritics and non-letter symbols.
        Keeps only Arabic letters and spaces.
        """
        if not text:
            return ""
        
        # Arabic diacritics (tashkeel) Unicode ranges
        arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')  # Most common diacritics + tatweel
        
        # Remove diacritics
        normalized = arabic_diacritics.sub('', text)
        
        # Keep only Arabic letters, spaces, and basic punctuation
        # Arabic letters range: \u0621-\u063A, \u0641-\u064A
        arabic_letters_and_space = re.compile(r'[^\u0621-\u063A\u0641-\u064A\s]')
        normalized = arabic_letters_and_space.sub('', normalized)
        
        # Clean up multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def extract_words_from_text(self, text):
        """Extract valid words from text"""
        if not text:
            return []
        
        words = text.split()
        return [word for word in words if len(word) >= self.min_word_length]
    
    def build_ayah_lookup_entry(self, ayah_data):
        """Create lookup entry for an ayah with both exact and normalized text"""
        normalized_text = self.normalize_arabic_text(ayah_data['ayah_text'])
        
        return {
            'surah_id': ayah_data['surah_id'],
            'surah_name': ayah_data['surah_name'],
            'ayah_id': ayah_data['ayah_id'],
            'ayah_text': ayah_data['ayah_text'],  # Exact text with tashkeel
            'normalized_text': normalized_text   # Text without tashkeel
        }
    
    def build_index_from_ayahs(self, ayah_json_path):
        """Build the inverse index from ayah data"""
        print(f"Loading ayahs from: {ayah_json_path}")
        
        with open(ayah_json_path, 'r', encoding='utf-8') as f:
            ayahs = json.load(f)
        
        print(f"Processing {len(ayahs)} ayahs...")
        
        for ayah in ayahs:
            # Create unique ayah identifier
            ayah_key = f"ayah_{ayah['surah_id']}_{ayah['ayah_id']}"
            
            # Build lookup entry (includes both exact and normalized text)
            lookup_entry = self.build_ayah_lookup_entry(ayah)
            self.ayah_lookup[ayah_key] = lookup_entry
            
            # Extract words from exact text
            exact_words = self.extract_words_from_text(ayah['ayah_text'])
            
            # Extract words from normalized text
            normalized_words = self.extract_words_from_text(lookup_entry['normalized_text'])
            
            # Index both exact and normalized words in the same index
            # This creates a unified searchable vocabulary
            all_words = set()
            
            # Add exact words
            for word in exact_words:
                all_words.add(word)
                self.stats['exact_words_added'] += 1
            
            # Add normalized words (avoid duplicates)
            for word in normalized_words:
                if word not in all_words:  # Only add if not already present as exact word
                    all_words.add(word)
                    self.stats['normalized_words_added'] += 1
            
            # Index all unique words to this ayah
            for word in all_words:
                self.word_index[word].add(ayah_key)
                self.stats['total_word_entries'] += 1
            
            self.stats['total_ayahs'] += 1
        
        # Finalize stats
        self.stats['unique_words'] = len(self.word_index)
        
        print("Index building completed!")
        self.print_stats()
    
    def print_stats(self):
        """Print indexing statistics"""
        print("\n=== Indexing Statistics ===")
        print(f"Total ayahs processed: {self.stats['total_ayahs']:,}")
        print(f"Unique words indexed (exact + normalized): {self.stats['unique_words']:,}")
        print(f"Total word->ayah mappings: {self.stats['total_word_entries']:,}")
        print(f"Exact words added: {self.stats['exact_words_added']:,}")
        print(f"Normalized words added: {self.stats['normalized_words_added']:,}")
        
        if self.stats['total_ayahs'] > 0:
            avg_words_per_ayah = self.stats['total_word_entries'] / self.stats['total_ayahs']
            print(f"Average word entries per ayah: {avg_words_per_ayah:.1f}")
    
    def save_index_to_files(self, output_dir="quran_index"):
        """Save the index and lookup data to files"""
        if not self.save_index:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving index to: {output_path}")
        
        # Convert defaultdict to regular dict and sets to lists for JSON serialization
        word_index_serializable = {
            word: list(ayah_ids) for word, ayah_ids in self.word_index.items()
        }
        
        # Save unified word index (contains both exact and normalized words)
        word_index_file = output_path / "quran_word_index.json"
        with open(word_index_file, 'w', encoding='utf-8') as f:
            json.dump(word_index_serializable, f, ensure_ascii=False, indent=2)
        print(f"Unified word index saved: {word_index_file}")
        
        # Save ayah lookup
        lookup_file = output_path / "quran_ayah_lookup.json"
        with open(lookup_file, 'w', encoding='utf-8') as f:
            json.dump(self.ayah_lookup, f, ensure_ascii=False, indent=2)
        print(f"Ayah lookup saved: {lookup_file}")
        
        # Save statistics
        stats_file = output_path / "index_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved: {stats_file}")
        
        return output_path
    
    def query_word(self, word, max_results=10):
        """
        Query the unified index for a specific word (for testing)
        Searches for both the word as-is and its normalized version
        """
        ayah_ids = set()
        
        # Search for the word as-is
        if word in self.word_index:
            ayah_ids.update(self.word_index[word])
        
        # Also search for the normalized version
        normalized_word = self.normalize_arabic_text(word)
        if normalized_word and normalized_word != word and normalized_word in self.word_index:
            ayah_ids.update(self.word_index[normalized_word])
        
        # Convert to list and limit results
        ayah_ids = list(ayah_ids)[:max_results]
        results = []
        
        for ayah_id in ayah_ids:
            if ayah_id in self.ayah_lookup:
                lookup_data = self.ayah_lookup[ayah_id]
                results.append({
                    'ayah_id': ayah_id,
                    'surah_ayah': f"{lookup_data['surah_id']}:{lookup_data['ayah_id']}",
                    'surah_name': lookup_data['surah_name'],
                    'ayah_text': lookup_data['ayah_text'],
                    'normalized_text': lookup_data['normalized_text']
                })
        
        return results


class QuranIndexLoader:
    """
    Loads the pre-built inverse index for fast querying.
    Supports both exact and normalized text searching.
    """
    
    def __init__(self, index_dir="quran_index"):
        self.index_dir = Path(index_dir)
        self.word_index = {}  # Unified index containing both exact and normalized words
        self.ayah_lookup = {}
        self.stats = {}
        
    def normalize_arabic_text(self, text):
        """Same normalization function as the builder"""
        if not text:
            return ""
        
        arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        normalized = arabic_diacritics.sub('', text)
        
        arabic_letters_and_space = re.compile(r'[^\u0621-\u063A\u0641-\u064A\s]')
        normalized = arabic_letters_and_space.sub('', normalized)
        
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
        
    def load_index(self):
        """Load the pre-built index from files"""
        try:
            # Load unified word index
            word_index_file = self.index_dir / "quran_word_index.json"
            with open(word_index_file, 'r', encoding='utf-8') as f:
                word_index_data = json.load(f)
                # Convert lists back to sets for faster lookup
                self.word_index = {
                    word: set(ayah_ids) for word, ayah_ids in word_index_data.items()
                }
            
            # Load ayah lookup
            lookup_file = self.index_dir / "quran_ayah_lookup.json"
            with open(lookup_file, 'r', encoding='utf-8') as f:
                self.ayah_lookup = json.load(f)
            
            # Load stats if available
            try:
                stats_file = self.index_dir / "index_stats.json"
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except FileNotFoundError:
                pass
            
            print(f"Index loaded successfully from: {self.index_dir}")
            print(f"Loaded {len(self.word_index):,} words and {len(self.ayah_lookup):,} ayahs")
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading index: {e}")
            return False
    
    def find_candidates(self, query_words, min_word_ratio=0.6):
        """
        Fast candidate retrieval using the unified index
        Automatically searches for both exact and normalized versions of each query word
        """
        if not query_words:
            return set()
        
        # Count how many words each ayah contains
        ayah_word_counts = defaultdict(int)
        
        for word in query_words:
            # Search for the word as-is
            if word in self.word_index:
                word_ayahs = self.word_index[word]
                for ayah_id in word_ayahs:
                    ayah_word_counts[ayah_id] += 1
            
            # Also search for normalized version (if different)
            normalized_word = self.normalize_arabic_text(word)
            if normalized_word and normalized_word != word and normalized_word in self.word_index:
                word_ayahs = self.word_index[normalized_word]
                for ayah_id in word_ayahs:
                    ayah_word_counts[ayah_id] += 1
        
        # Filter ayahs that have at least min_word_ratio of the query words
        min_required_words = max(1, int(len(query_words) * min_word_ratio))
        candidates = set()
        
        for ayah_id, word_count in ayah_word_counts.items():
            if word_count >= min_required_words:
                candidates.add(ayah_id)
        
        return candidates
    
    def search_ayahs(self, query_words, max_results=10):
        """
        Search for ayahs containing the query words
        Automatically searches both exact and normalized versions
        """
        candidates = self.find_candidates(query_words)
        
        # Sort candidates by number of matching words (simple relevance)
        ayah_scores = defaultdict(int)
        
        for word in query_words:
            # Count matches for word as-is
            if word in self.word_index:
                for ayah_id in self.word_index[word]:
                    if ayah_id in candidates:
                        ayah_scores[ayah_id] += 1
            
            # Count matches for normalized version (if different)
            normalized_word = self.normalize_arabic_text(word)
            if normalized_word and normalized_word != word and normalized_word in self.word_index:
                for ayah_id in self.word_index[normalized_word]:
                    if ayah_id in candidates:
                        ayah_scores[ayah_id] += 1
        
        # Get top results
        sorted_ayahs = sorted(ayah_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for ayah_id, score in sorted_ayahs[:max_results]:
            if ayah_id in self.ayah_lookup:
                ayah_data = self.ayah_lookup[ayah_id]
                results.append({
                    'ayah_id': ayah_id,
                    'surah_ayah': f"{ayah_data['surah_id']}:{ayah_data['ayah_id']}",
                    'surah_name': ayah_data['surah_name'],
                    'ayah_text': ayah_data['ayah_text'],
                    'normalized_text': ayah_data['normalized_text'],
                    'matching_words': score
                })
        
        return results