# Span Correction - SubTask C

The corrector is split into 3 main components:

### 1. `matcher.py` - Core Matching Logic
- **QuranHadithSpanMatcher**: Main class for span matching
- Contains text normalization, indexing, and similarity calculation
- Supports cross-ayah matching for Quran
- Integrates with HuggingFace re-ranker

### 2. `reranker.py` - Re-ranking model
- **HuggingFaceReRanker**: Semantic re-ranking using transformer models
- Uses BAAI/bge-reranker-v2-m3 by default
- Provides weighted combination of original and re-ranker scores

### 3. `span_corrector.py` - Processing & Submission
- **SpanProcessor**: Handles file I/O, XML processing, and tsv generation
- Main processing pipeline for test datasets
- Stores detailed span information for analysis

## Usage

### 1st Usage (Without Re-ranker)
```python
from matcher import QuranHadithSpanMatcher
from span_processor import SpanProcessor

# Initialize without HF re-ranker
matcher = QuranHadithSpanMatcher(
    quran_index_dir="quran_index",
    hadith_index_dir="hadith_index",
    use_hf_reranker=False,
    verbose=True
)

processor = SpanProcessor(matcher, verbose=True)
```

### 2nd Usage (With Re-ranker)
```python
# Initialize with HF re-ranker
matcher = QuranHadithSpanMatcher(
    quran_index_dir="quran_index",
    hadith_index_dir="hadith_index",
    use_hf_reranker=True,
    reranker_model="BAAI/bge-reranker-v2-m3",
    verbose=True
)

processor = SpanProcessor(matcher, verbose=True)
```

## Running the System

### Method 1: Using the main script
```bash
python span_processor.py
```