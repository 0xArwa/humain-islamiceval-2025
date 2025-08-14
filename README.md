# HUMAIN @ Islamic Shared Task 2025

This repository contains our proposed system for the **IslamicEval 2025 Shared Task**, focusing on Islamic text processing for Quran and Hadith span detection, verification, and correction.

## Task Overview

### **Subtask A: Span Detection** 🔍
- **Goal**: Automatically detect and extract Quranic verses and Hadith spans from AI-generated responses
- **Input**: LLM model responses in XML format
- **Output**: TSV file with detected spans and their positions

### **Subtask B: Span Verification** ✅
- **Goal**: Verify the correctness of detected Quranic and Hadith spans
- **Input**: Detected spans from Subtask A
- **Output**: Correct/Incorrect labels for each span

### **Subtask C: Span Correction** 🔧
- **Goal**: Correct incorrect spans by finding the most similar correct spans
- **Input**: Incorrect spans from Subtask B
- **Output**: Corrected spans or error message

## 🚀 Quick Start

### Prerequisites
```bash
# Install all dependencies for all subtasks
pip install -r requirements.txt

```

### Unified Interface
Use the main script to run any subtask:

```bash
# Run Subtask A (Span Detection)
python main.py --task A --mode dev

# Run Subtask B (Span Verification)  
python main.py --task B --tsv dev_data.tsv --xml questions.xml

# Run Subtask C (Span Correction)
python main.py --task C --input dev_data.tsv --output corrected.tsv

# Run all tasks sequentially
python main.py --task ALL --mode dev
```

## 📁 Repository Structure

```
islamic_shared_task_2025/
├── main.py                 # Unified entry point for all subtasks
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── datasets/              # Reference datasets
│   ├── quranic_verses.json
│   ├── six_hadith_books.json
│   └── test files...
├── subtask_A/            # Span Detection
│   ├── main.py           # Subtask A entry point
│   ├── span_detection.py # Core detection logic
│   ├── data_processor.py # Data processing utilities
│   ├── config.py         # Configuration management
│   └── submission_A/     # Generated submissions
├── subtask_B/            # Span Verification  
│   ├── span_checker.py   # Main verification script
│   ├── quran_verification.py # Quran verification logic
│   ├── hadith_verification.py # Hadith verification logic
│   ├── multi_verse_detector.py # Multi-verse detection
│   └── submission_B/     # Generated submissions
└── subtask_C/            # Span Correction
    ├── span_corrector.py # Main correction script
    ├── matcher.py        # Core matching logic
    ├── reranker.py       # Semantic re-ranking
    └── submission_C/     # Generated submissions
```

## 🔧 Detailed Usage

> **📖 For advanced configurations, threshold tuning, and detailed parameter explanations, please refer to each subtask's individual README file:**
> - [`subtask_A/README.md`](subtask_A/README.md) - Span Detection configuration and prompts
> - [`subtask_B/README.md`](subtask_B/README.md) - Verification thresholds and optimization
> - [`subtask_C/README.md`](subtask_C/README.md) - Correction algorithms and re-ranking models


## Acknowledgement

We thank the organizers of the IslamicEval 2025 Shared Task for providing the datasets.

This work was conducted by HUMAIN research team. We acknowledge the open-source community for the tools and libraries that made this work possible.