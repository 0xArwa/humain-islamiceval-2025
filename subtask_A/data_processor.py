import json
import datetime
import xml.etree.ElementTree as ET
import pandas as pd
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import Config
from span_detection import SpanDetection

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.span_service = SpanDetection(config)
    
    def load_xml_data(self, xml_path: str) -> ET.Element:
        """Load and parse XML data"""
        with open(xml_path, encoding='utf-8') as f:
            lines = [line for line in f]
        
        xml_content = ''.join(lines)
        # Add root element if not present
        root = ET.fromstring('<ROOT>' + xml_content + '</ROOT>')
        return root
    
    def load_tsv_data(self, tsv_path: str) -> pd.DataFrame:
        """Load TSV data"""
        return pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
    
    def process_single_question(self, question: ET.Element) -> Tuple[str, str, str, str]:
        """Process a single question element"""
        qid = question.find('ID').text
        text = question.find('Text').text
        response = question.find('Response').text
        
        try:
            llm_prediction = self.span_service.detect_spans_from_text(response)
            return (qid, text, response, llm_prediction)
        except Exception as e:
            print(f"Error processing Question ID: {qid}, Error: {e}")
            return (qid, text, response, "")
    
    def process_questions_sequential(self, root: ET.Element, existing_predictions: List = None) -> List[Tuple[str, str, str, str]]:
        """Process questions sequentially"""
        predictions = existing_predictions or []
        processed_ids = {pred[0] for pred in predictions}
        
        for q in tqdm(root.findall('Question'), desc="Processing Questions"):
            qid = q.find('ID').text
            
            if qid in processed_ids:
                print(f"Skipping already processed Question ID: {qid}")
                continue
            
            result = self.process_single_question(q)
            if result[3]:  # Only add if we got a valid prediction
                predictions.append(result)
        
        return predictions
    
    def process_questions_parallel(self, root: ET.Element, existing_predictions: List = None) -> List[Tuple[str, str, str, str]]:
        """Process questions in parallel"""
        predictions = existing_predictions or []
        processed_ids = {pred[0] for pred in predictions}
        
        questions = [q for q in root.findall('Question')]
        results_ordered = [None] * len(questions)
        
        def process_question_with_index(idx_q):
            idx, q = idx_q
            qid = q.find('ID').text
            
            if qid in processed_ids:
                print(f"Skipping already processed Question ID: {qid}")
                return None
            
            result = self.process_single_question(q)
            return (idx, result) if result[3] else None
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(process_question_with_index, (i, q)) 
                      for i, q in enumerate(questions)]
            
            for future in tqdm(futures, desc="Processing Questions (parallel)"):
                result = future.result()
                if result is not None:
                    idx, prediction = result
                    results_ordered[idx] = prediction
        
        # Filter out None values and extend predictions
        predictions.extend([pred for pred in results_ordered if pred is not None])
        return predictions
    
    def process_questions(self, xml_path: str, existing_predictions: List = None) -> List[Tuple[str, str, str, str]]:
        """Process questions using configured method"""
        root = self.load_xml_data(xml_path)
        
        if self.config.enable_multiprocessing:
            return self.process_questions_parallel(root, existing_predictions)
        else:
            return self.process_questions_sequential(root, existing_predictions)
    
    def save_predictions(self, predictions: List[Tuple[str, str, str, str]], filename_prefix: str = "LLM_predictions") -> str:
        """Save predictions to JSON file with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{self.config.environment}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        
        return filename
    
    def predictions_to_results(self, predictions: List[Tuple[str, str, str, str]]) -> List[Dict[str, Any]]:
        """Convert predictions to results format"""
        results = []
        
        for prediction in predictions:
            qid, text, response, construction = prediction
            tags_spans = self.span_service.get_tags_spans_from_construction(response, construction)
            annotation_id = 1
            
            for span, tag, start, end in tags_spans:
                try:
                    tag_type = 'Ayah' if tag[0][0] == 'ق' else 'Hadith'
                except Exception:
                    tag_type = 'Ayah'
                
                results.append({
                    "Question_ID": qid,
                    "Annotation_ID": annotation_id,
                    "Span_Type": tag_type,
                    "Span_Start": start,
                    "Span_End": end,
                    "Span_Text": span
                })
                annotation_id += 1
            
            if len(tags_spans) == 0:
                results.append({
                    "Question_ID": qid,
                    "Annotation_ID": 0,
                    "Span_Type": "No_Spans",
                    "Span_Start": 0,
                    "Span_End": 0,
                    "Span_Text": ""
                })
        
        return results
    
    def save_results_to_tsv(self, results: List[Dict[str, Any]], filename_prefix: str = "subtaskA_results") -> str:
        """Save results to TSV file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.environment}_{filename_prefix}_{timestamp}.tsv"
        
        results_df = pd.DataFrame(results, columns=[
            "Question_ID", "Annotation_ID", "Span_Start", "Span_End", "Span_Type", "Span_Text"
        ])
        
        results_df.to_csv(filename, sep='\t', index=False, encoding='utf-8', header=False)
        return filename
    
    def run_full_pipeline(self) -> Tuple[str, str]:
        """Run the complete processing pipeline"""
        xml_path = self.config.get_dataset_xml()
        
        print(f"Processing {xml_path} in {self.config.environment} mode...")
        
        # Process questions
        predictions = self.process_questions(xml_path)
        
        # Save predictions
        pred_file = self.save_predictions(predictions)
        print(f"Predictions saved to: {pred_file}")
        
        # Convert to results
        results = self.predictions_to_results(predictions)
        
        # Save results
        results_file = self.save_results_to_tsv(results)
        print(f"Results saved to: {results_file}")
        
        return pred_file, results_file