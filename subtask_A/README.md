# Span Detection - SubTask A

Utilizing LLMs for identifying Quran and Hadith references in Arabic text using GPT models through Azure OpenAI endpoint and VLLM for guided decoding.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Update the `.env` file with your Azure OpenAI credentials

3. **Run span detection:**
   ```bash
   python main.py --mode dev
   ```

4. **Run VLLM guided decoding (requires local VLLM server):**
   ```bash
   python vllm_guided_decoding.py
   ```

## Files

- `main.py` - Main application with GPT-based span detection
- `config.py` - Azure OpenAI configuration
- `span_detection.py` - Core span detection logic
- `vllm_guided_decoding.py` - VLLM guided decoding implementation