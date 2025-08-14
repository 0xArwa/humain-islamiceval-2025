import json
from collections import defaultdict
from pathlib import Path

class HadithInverseIndexBuilder:
    """
    Builds an optimized inverse index for Hadith text matching.
    Indexes both hadithTxt and Matn fields for comprehensive search.
    """
    
    def __init__(self, min_word_length=1, save_index=True):
        self.min_word_length = min_word_length
        self.save_index = save_index
        
        # Main inverse index: word -> set of hadith_ids
        self.word_index = defaultdict(set)
        
        # Hadith lookup: hadith_id -> hadith_data
        self.hadith_lookup = {}
        
        # Statistics
        self.stats = {
            'total_hadiths': 0,
            'unique_words': 0,
            'total_word_entries': 0,
            'hadiths_with_matn': 0,
            'hadiths_with_hadith_txt': 0
        }
    
    def extract_words_from_text(self, text):
        """Extract valid words from hadith text"""
        if not text:
            return []
        
        # Clean up text - remove extra whitespace and newlines
        cleaned_text = ' '.join(text.split())
        words = cleaned_text.split()
        return [word for word in words if len(word) >= self.min_word_length]
    
    def build_hadith_lookup_entry(self, hadith_data):
        """Create lookup entry for a hadith"""
        return {
            'hadithID': hadith_data['hadithID'],
            'BookID': hadith_data.get('BookID'),
            'title': hadith_data.get('title', ''),
            'hadithTxt': hadith_data.get('hadithTxt', ''),
            'Matn': hadith_data.get('Matn', '')
        }
    
    def build_index_from_hadiths(self, hadith_json_path):
        """Build the inverse index from hadith data"""
        print(f"Loading hadiths from: {hadith_json_path}")
        
        with open(hadith_json_path, 'r', encoding='utf-8') as f:
            hadiths = json.load(f)
        
        print(f"Processing {len(hadiths)} hadiths...")
        
        for hadith in hadiths:
            # Create unique hadith identifier
            hadith_key = f"hadith_{hadith['hadithID']}"
            
            # Extract words from both hadithTxt and Matn
            all_words = set()  # Use set to avoid duplicate words from same hadith
            
            # Process hadithTxt
            if hadith.get('hadithTxt'):
                hadith_txt_words = self.extract_words_from_text(hadith['hadithTxt'])
                all_words.update(hadith_txt_words)
                self.stats['hadiths_with_hadith_txt'] += 1
            
            # Process Matn
            if hadith.get('Matn'):
                matn_words = self.extract_words_from_text(hadith['Matn'])
                all_words.update(matn_words)
                self.stats['hadiths_with_matn'] += 1
            
            # Build lookup entry
            lookup_entry = self.build_hadith_lookup_entry(hadith)
            self.hadith_lookup[hadith_key] = lookup_entry
            
            # Index all unique words for this hadith
            for word in all_words:
                self.word_index[word].add(hadith_key)
                self.stats['total_word_entries'] += 1
            
            self.stats['total_hadiths'] += 1
        
        # Finalize stats
        self.stats['unique_words'] = len(self.word_index)
        
        print("Index building completed!")
        self.print_stats()
    
    def print_stats(self):
        """Print indexing statistics"""
        print("\n=== Indexing Statistics ===")
        print(f"Total hadiths processed: {self.stats['total_hadiths']:,}")
        print(f"Hadiths with hadithTxt: {self.stats['hadiths_with_hadith_txt']:,}")
        print(f"Hadiths with Matn: {self.stats['hadiths_with_matn']:,}")
        print(f"Unique words indexed: {self.stats['unique_words']:,}")
        print(f"Total word->hadith mappings: {self.stats['total_word_entries']:,}")
        
        if self.stats['total_hadiths'] > 0:
            avg_words_per_hadith = self.stats['total_word_entries'] / self.stats['total_hadiths']
            print(f"Average words per hadith: {avg_words_per_hadith:.1f}")
    
    def save_index_to_files(self, output_dir="hadith_index"):
        """Save the index and lookup data to files"""
        if not self.save_index:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving index to: {output_path}")
        
        # Convert defaultdict to regular dict and sets to lists for JSON serialization
        word_index_serializable = {
            word: list(hadith_ids) for word, hadith_ids in self.word_index.items()
        }
        
        # Save word index
        word_index_file = output_path / "hadith_word_index.json"
        with open(word_index_file, 'w', encoding='utf-8') as f:
            json.dump(word_index_serializable, f, ensure_ascii=False, indent=2)
        print(f"Word index saved: {word_index_file}")
        
        # Save hadith lookup
        lookup_file = output_path / "hadith_lookup.json"
        with open(lookup_file, 'w', encoding='utf-8') as f:
            json.dump(self.hadith_lookup, f, ensure_ascii=False, indent=2)
        print(f"Hadith lookup saved: {lookup_file}")
        
        # Save statistics
        stats_file = output_path / "index_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved: {stats_file}")
        
        return output_path
    
    def query_word(self, word, max_results=10):
        """Query the index for a specific word (for testing)"""
        if word in self.word_index:
            hadith_ids = list(self.word_index[word])[:max_results]
            results = []
            
            for hadith_id in hadith_ids:
                if hadith_id in self.hadith_lookup:
                    lookup_data = self.hadith_lookup[hadith_id]
                    results.append({
                        'hadith_id': hadith_id,
                        'hadithID': lookup_data['hadithID'],
                        'BookID': lookup_data['BookID'],
                        'title': lookup_data['title'],
                        'hadithTxt': lookup_data['hadithTxt'][:200] + '...' if len(lookup_data['hadithTxt']) > 200 else lookup_data['hadithTxt'],
                        'Matn': lookup_data['Matn'][:200] + '...' if lookup_data.get('Matn') and len(lookup_data['Matn']) > 200 else lookup_data.get('Matn', '')
                    })
            
            return results
        
        return []


class HadithIndexLoader:
    """
    Loads the pre-built inverse index for fast hadith querying.
    """
    
    def __init__(self, index_dir="hadith_index"):
        self.index_dir = Path(index_dir)
        self.word_index = {}
        self.hadith_lookup = {}
        self.stats = {}
        
    def load_index(self):
        """Load the pre-built index from files"""
        try:
            # Load word index
            word_index_file = self.index_dir / "hadith_word_index.json"
            with open(word_index_file, 'r', encoding='utf-8') as f:
                word_index_data = json.load(f)
                # Convert lists back to sets for faster lookup
                self.word_index = {
                    word: set(hadith_ids) for word, hadith_ids in word_index_data.items()
                }
            
            # Load hadith lookup
            lookup_file = self.index_dir / "hadith_lookup.json"
            with open(lookup_file, 'r', encoding='utf-8') as f:
                self.hadith_lookup = json.load(f)
            
            # Load stats if available
            try:
                stats_file = self.index_dir / "index_stats.json"
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except FileNotFoundError:
                pass
            
            print(f"Index loaded successfully from: {self.index_dir}")
            print(f"Loaded {len(self.word_index):,} words and {len(self.hadith_lookup):,} hadiths")
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading index: {e}")
            return False
    
    def find_candidates(self, query_words, min_word_ratio=0.6):
        """Fast candidate retrieval using the loaded index"""
        if not query_words:
            return set()
        
        # Count how many words each hadith contains
        hadith_word_counts = defaultdict(int)
        
        for word in query_words:
            if word in self.word_index:
                word_hadiths = self.word_index[word]
                for hadith_id in word_hadiths:
                    hadith_word_counts[hadith_id] += 1
        
        # Filter hadiths that have at least min_word_ratio of the query words
        min_required_words = max(1, int(len(query_words) * min_word_ratio))
        candidates = set()
        
        for hadith_id, word_count in hadith_word_counts.items():
            if word_count >= min_required_words:
                candidates.add(hadith_id)
        
        return candidates
    
    def search_hadiths(self, query_words, max_results=10):
        """Search for hadiths containing the query words"""
        candidates = self.find_candidates(query_words)
        
        # Sort candidates by number of matching words (simple relevance)
        hadith_scores = defaultdict(int)
        for word in query_words:
            if word in self.word_index:
                for hadith_id in self.word_index[word]:
                    if hadith_id in candidates:
                        hadith_scores[hadith_id] += 1
        
        # Get top results
        sorted_hadiths = sorted(hadith_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for hadith_id, score in sorted_hadiths[:max_results]:
            if hadith_id in self.hadith_lookup:
                hadith_data = self.hadith_lookup[hadith_id]
                results.append({
                    'hadith_id': hadith_id,
                    'hadithID': hadith_data['hadithID'],
                    'BookID': hadith_data['BookID'],
                    'title': hadith_data['title'],
                    'hadithTxt': hadith_data['hadithTxt'],
                    'Matn': hadith_data['Matn'],
                    'matching_words': score
                })
        
        return results