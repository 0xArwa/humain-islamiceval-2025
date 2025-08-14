import requests
from typing import Dict, Any, List, Tuple
from span_detection import SpanDetection
from config import Config
import xml.etree.ElementTree as ET
import json
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Schema for guided decoding - JSON schema for spans array
SPANS_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["q", "h"],
                        "description": "Type of span: 'q' for Quran, 'h' for Hadith"
                    },
                    "text": {
                        "type": "string",
                        "description": "The actual text of the span"
                    },
                    "start": {
                        "type": "integer",
                        "description": "Start character index"
                    },
                    "end": {
                        "type": "integer", 
                        "description": "End character index"
                    }
                },
                "required": ["type", "text", "start", "end"]
            }
        }
    },
    "required": ["spans"]
}

VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

# system prompt for structured JSON output
STRUCTURED_SYSTEM_PROMPT = """You are an expert assistant specializing in identifying Islamic religious texts. Your task is to detect and extract spans of Quranic verses and Hadith from Arabic text.

Instructions:
1. Identify all spans that contain Quranic verses or Hadith sayings
2. For each span found, determine its exact character positions in the original text
3. Use type "q" for Quranic verses and "h" for Hadith sayings
4. Extract only the actual religious text, not references, verse numbers, or commentary
5. Return results as a JSON object with a "spans" array

Output Format:
{
  "spans": [
    {
      "type": "q",
      "text": "النص القرآني هنا",
      "start": 0,
      "end": 15
    },
    {
      "type": "h", 
      "text": "نص الحديث هنا",
      "start": 20,
      "end": 35
    }
  ]
}

Examples:

Input: "إِنَّا أَنْزَلْنَاهُ فِي لَيْلَةِ الْقَدْرِ والله أعلم"
Output: {"spans": [{"type": "q", "text": "إِنَّا أَنْزَلْنَاهُ فِي لَيْلَةِ الْقَدْرِ", "start": 0, "end": 29}]}

Input: "المؤمن القوي خير وأحب إلى الله من المؤمن الضعيف "
Output: {"spans": [{"type": "h", "text": "المؤمن القوي خير وأحب إلى الله من المؤمن الضعيف", "start": 0, "end": 46}]}

If no spans are found, return: {"spans": []}
"""

# send a request to the vLLM endpoint with guided decoding
def get_guided_decoding(prompt: str, schema: Dict[str, Any], max_tokens: int = 1000) -> Dict[str, Any]:
    payload = {
        "model": "allam",
        "messages": [
            {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "guided_json": schema,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.98
    }
    
    try:
        response = requests.post(VLLM_ENDPOINT, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling vLLM endpoint: {e}")
        return {"choices": [{"message": {"content": '{"spans": []}'}}]}

class VLLMSpanDetection(SpanDetection):
    def __init__(self, config: Config):
        super().__init__(config)

    def detect_spans_with_vllm(self, text: str) -> Dict[str, Any]:
        """Detect spans using vLLM with guided decoding and return structured JSON"""
        # Clean the input text
        preprocessed_text = text.replace('[', '{').replace(']', '}').replace('|', ',')
        
        # Use guided decoding with JSON schema
        response = get_guided_decoding(preprocessed_text, SPANS_SCHEMA, max_tokens=2048)
        
        try:
            # Extract content from vLLM response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", '{"spans": []}')
            
            # Parse JSON content
            if isinstance(content, str):
                parsed_content = json.loads(content)
            else:
                parsed_content = content
                
            return parsed_content
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing vLLM response: {e}")
            print(f"Raw response: {response}")
            return {"spans": []}

    def get_tags_spans_from_vllm(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract tags and spans from vLLM response and convert to expected format"""
        try:
            # Get structured JSON response
            json_response = self.detect_spans_with_vllm(text)
            spans_data = json_response.get("spans", [])
            
            # Convert to expected tuple format: (span_text, tag, start, end)
            result_spans = []
            for span in spans_data:
                span_text = span.get("text", "")
                span_type = span.get("type", "")
                start_idx = span.get("start", 0)
                end_idx = span.get("end", 0)
                
                # Convert type to tag format expected by original code
                tag = "ق" if span_type == "q" else "ح" if span_type == "h" else ""
                
                # Validate span indices and text
                if (0 <= start_idx < end_idx <= len(text) and 
                    span_text.strip() and tag):
                    result_spans.append((span_text, tag, start_idx, end_idx))
                    
            return result_spans
            
        except Exception as e:
            print(f"Error in get_tags_spans_from_vllm: {e}")
            return []

    def validate_spans(self, text: str, spans: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, int, int]]:
        """Validate that extracted spans actually exist in the original text"""
        validated_spans = []
        
        for span_text, tags, start, end in spans:
            try:
                # Check if the span indices are valid
                if 0 <= start < end <= len(text):
                    extracted_text = text[start:end]
                    # Allow for some whitespace differences and clean comparison
                    if self._normalize_text(extracted_text) == self._normalize_text(span_text):
                        validated_spans.append((span_text, tags, start, end))
                    else:
                        # Try to find the span text in nearby positions
                        corrected_span = self._find_corrected_span(text, span_text, start, end)
                        if corrected_span:
                            validated_spans.append(corrected_span)
                        else:
                            print(f"Warning: Span text mismatch. Expected: '{span_text}', Got: '{extracted_text}'")
                else:
                    print(f"Warning: Invalid span indices [{start}:{end}] for text of length {len(text)}")
            except Exception as e:
                print(f"Error validating span: {e}")
                continue
                
        return validated_spans
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and diacritics"""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized
    
    def _find_corrected_span(self, text: str, span_text: str, start: int, end: int) -> Tuple[str, str, int, int]:
        """Try to find the correct span position if initial indices are wrong"""
        normalized_span = self._normalize_text(span_text)
        
        # Search in a window around the original position
        search_window = 50
        search_start = max(0, start - search_window)
        search_end = min(len(text), end + search_window)
        search_area = text[search_start:search_end]
        
        # Try to find the span text
        span_start = search_area.find(span_text)
        if span_start != -1:
            corrected_start = search_start + span_start
            corrected_end = corrected_start + len(span_text)
            return (span_text, "ق", corrected_start, corrected_end)  # Default to Quran tag
            
        return None

class VLLMDataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.span_service = VLLMSpanDetection(config)
    
    def load_xml_data(self, xml_path: str) -> ET.Element:
        """Load and parse XML data"""
        with open(xml_path, encoding='utf-8') as f:
            lines = [line for line in f]
        
        xml_content = ''.join(lines)
        # Add root element if not present
        root = ET.fromstring('<ROOT>' + xml_content + '</ROOT>')
        return root
    
    def process_single_question(self, question: ET.Element) -> Tuple[str, str, str, List[Tuple[str, str, int, int]]]:
        """Process a single question element with vLLM"""
        qid = question.find('ID').text
        text = question.find('Text').text
        response = question.find('Response').text
        
        try:
            # Use vLLM guided decoding to extract spans
            spans = self.span_service.get_tags_spans_from_vllm(response)
            validated_spans = self.span_service.validate_spans(response, spans)
            return (qid, text, response, validated_spans)
        except Exception as e:
            print(f"Error processing Question ID: {qid}, Error: {e}")
            return (qid, text, response, [])
    
    def process_questions_sequential(self, root: ET.Element) -> List[Tuple[str, str, str, List[Tuple[str, str, int, int]]]]:
        """Process questions sequentially"""
        results = []
        
        for q in tqdm(root.findall('Question'), desc="Processing Questions"):
            result = self.process_single_question(q)
            results.append(result)
        
        return results
    
    def process_questions_parallel(self, root: ET.Element) -> List[Tuple[str, str, str, List[Tuple[str, str, int, int]]]]:
        """Process questions in parallel"""
        questions = [q for q in root.findall('Question')]
        results_ordered = [None] * len(questions)
        
        def process_question_with_index(idx_q):
            idx, q = idx_q
            result = self.process_single_question(q)
            return (idx, result)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(process_question_with_index, (i, q)) 
                      for i, q in enumerate(questions)]
            
            for future in tqdm(futures, desc="Processing Questions (parallel)"):
                idx, result = future.result()
                results_ordered[idx] = result
        
        return results_ordered
    
    def process_questions(self, xml_path: str) -> List[Tuple[str, str, str, List[Tuple[str, str, int, int]]]]:
        """Process questions using configured method"""
        root = self.load_xml_data(xml_path)
        
        if self.config.enable_multiprocessing:
            return self.process_questions_parallel(root)
        else:
            return self.process_questions_sequential(root)
    
    def results_to_tsv_format(self, results: List[Tuple[str, str, str, List[Tuple[str, str, int, int]]]]) -> List[Dict[str, Any]]:
        """Convert results to TSV format"""
        tsv_results = []
        
        for result in results:
            qid, text, response, spans = result
            annotation_id = 1
            
            for span_text, tag, start, end in spans:
                try:
                    tag_type = 'Ayah' if tag == 'ق' else 'Hadith'
                except Exception:
                    tag_type = 'Ayah'
                
                tsv_results.append({
                    "Question_ID": qid,
                    "Annotation_ID": annotation_id,
                    "Span_Type": tag_type,
                    "Span_Start": start,
                    "Span_End": end,
                    "Span_Text": span_text
                })
                annotation_id += 1
            
            if len(spans) == 0:
                tsv_results.append({
                    "Question_ID": qid,
                    "Annotation_ID": 0,
                    "Span_Type": "No_Spans",
                    "Span_Start": 0,
                    "Span_End": 0,
                    "Span_Text": ""
                })
        
        return tsv_results
    
    def save_results_to_tsv(self, results: List[Dict[str, Any]], filename_prefix: str = "vllm_subtaskA_results") -> str:
        """Save results to TSV file"""
        
        filename = f"{self.config.environment}_{filename_prefix}.tsv"
        
        results_df = pd.DataFrame(results, columns=[
            "Question_ID", "Annotation_ID", "Span_Start", "Span_End", "Span_Type", "Span_Text"
        ])
        
        results_df.to_csv(filename, sep='\t', index=False, encoding='utf-8', header=False)
        return filename
    
    def run_full_pipeline(self) -> str:
        """Run the complete vLLM processing pipeline"""
        xml_path = self.config.get_dataset_xml()
        
        print(f"Processing {xml_path} with vLLM guided decoding in {self.config.environment} mode...")
        
        # Process questions
        results = self.process_questions(xml_path)
        
        # Convert to TSV format
        tsv_results = self.results_to_tsv_format(results)
        
        # Save results
        results_file = self.save_results_to_tsv(tsv_results)
        print(f"Results saved to: {results_file}")
        
        return results_file

if __name__ == "__main__":
    config = Config.from_env()
    processor = VLLMDataProcessor(config)
    
    try:
        results_file = processor.run_full_pipeline()
        print(f"vLLM guided decoding processing completed successfully!")
        print(f"TSV results saved to: {results_file}")
    except Exception as e:
        print(f"Error during processing: {e}")