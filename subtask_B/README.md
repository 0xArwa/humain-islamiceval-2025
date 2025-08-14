# SubTask 1B - Quran and Hadith Verification

This folder contains a complete Islamic text verification system for SubTask B - verifying Quranic and Hadith text spans.


### Files
- `span_checker.py`: Unified evaluation script for both Quran and Hadith verification
- `quran_verification.py`: Optimized Quran verse verification with indexing
- `hadith_verification.py`: Hadith text verification with adaptive matching
- `verse_span_validator.py`: Validates and corrects Arabic text spans
- `diacritics_checker.py`: Verifies Arabic diacritics (tashkeel) accuracy
- `multi_verse_detector.py`: Detects multi-verse spans with smart separators

## Usage

### Quick Start
```bash
# Run complete evaluation on test data
python span_checker.py --tsv dev.tsv --xml dev_questions.xml --types Ayah Hadith

```

## Configuration

Key parameters in `span_checker.py`:
- `QURAN_CONFIG`: Similarity thresholds and matching strategies for Quran
- `HADITH_CONFIG`: Verification parameters for Hadith
- `USE_DIACRITICS_CHECKING`: Enable/disable diacritics validation