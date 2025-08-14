import csv
import xml.etree.ElementTree as ET
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from matcher import QuranHadithSpanMatcher


class SpanCorrector:
    """
    Main processor for handling span correction tasks
    Manages file I/O, XML processing, and submission generation
    """
    
    def __init__(self, matcher: QuranHadithSpanMatcher, verbose: bool = False):
        self.matcher = matcher
        self.verbose = verbose
    
    def determine_search_source(self, span_type: str) -> str:
        """Determine search source based on span type"""
        span_type_lower = span_type.lower()
        
        if 'ayah' in span_type_lower:
            return 'quran'
        elif 'hadith' in span_type_lower:
            return 'hadith'
        else:
            raise ValueError(f"Unknown span type: {span_type}")

    
    def extract_full_span_from_xml(self, xml_file: str, question_id: str, span_start: int, span_end: int) -> Optional[str]:
        """Extract full span text from XML file using question ID and span positions"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Find question by ID
            for question in root.findall('.//Question'):
                q_id = question.get('ID')
                if q_id == question_id:
                    
                    # Find body text
                    body = question.find('.//Body')
                    if body is not None and body.text:
                        body_text = body.text
                        
                        # Validate span positions
                        if 0 <= span_start < span_end <= len(body_text):
                            extracted_span = body_text[span_start:span_end]
                            return extracted_span.strip()
                        else:
                            if self.verbose:
                                print(f"    Invalid span positions for Q{question_id}: start={span_start}, end={span_end}, text_len={len(body_text)}")
                    break
        
        except Exception as e:
            if self.verbose:
                print(f"    Error extracting span from XML: {e}")
        
        return None
    
    def save_json(self, data: List[Dict], output_file: str):
        """Save data to JSON file with proper formatting"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_submission_tsv(self, data: List[Dict], output_file: str):
        """Save submission data to TSV file with proper string formatting"""
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header
            writer.writerow(['Sequence_ID', 'Correction'])
            
            # Write data with explicit string conversion
            for row in data:
                seq_id = str(row['Sequence_ID']).strip()  # Ensure string format
                correction = str(row['Correction']).strip()  # Ensure string format
                writer.writerow([seq_id, correction])
    
    
    def print_submission_stats(self, submission_data: List[Dict]):
        """Print statistics about the submission"""
        total = len(submission_data)
        corrections = sum(1 for item in submission_data if item['Correction'] != 'خطأ')
        errors = total - corrections
        
        print(f"\n{'='*60}")
        print(f"SUBMISSION STATISTICS")
        print(f"{'='*60}")
        print(f"Total sequences: {total}")
        print(f"Corrections found: {corrections} ({corrections/total*100:.1f}%)")
        print(f"No corrections (خطأ): {errors} ({errors/total*100:.1f}%)")
        print(f"{'='*60}")
    
    def print_spans_stats(self, spans_data: List[Dict]):
        """Print statistics about the extracted spans"""
        if not spans_data:
            return
        
        total = len(spans_data)
        xml_extractions = sum(1 for span in spans_data if span.get('Extraction_Source') == 'XML')
        corrections_found = sum(1 for span in spans_data if span.get('System_Correction') != 'خطأ')
        
        # Source type breakdown
        quran_spans = sum(1 for span in spans_data if 'Quran_Details' in span)
        hadith_spans = sum(1 for span in spans_data if 'Hadith_Details' in span)
        
        # Cross-ayah matches
        cross_ayah_matches = sum(1 for span in spans_data 
                                if span.get('Match_Details', {}).get('is_cross_ayah', False))
        
        print(f"\n{'='*60}")
        print(f"SPANS PROCESSING STATISTICS")
        print(f"{'='*60}")
        print(f"Total spans processed: {total}")
        print(f"XML extractions: {xml_extractions} ({xml_extractions/total*100:.1f}%)")
        print(f"Corrections found: {corrections_found} ({corrections_found/total*100:.1f}%)")
        print(f"Quran matches: {quran_spans}")
        print(f"Hadith matches: {hadith_spans}")
        print(f"Cross-ayah matches: {cross_ayah_matches}")
        
        # Average confidence
        confidences = [span.get('Confidence', 0.0) for span in spans_data if span.get('Confidence', 0.0) > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"Average confidence: {avg_confidence:.3f}")
        
        print(f"{'='*60}")
    
    def process_test_tsv_file(self, tsv_file: str, xml_file: Optional[str] = None, 
                             output_file: str = "submission.tsv", 
                             output_spans_file: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """Process test TSV file and create submission format output with extracted spans storage"""
        submission_data = []
        test_spans_with_corrections = []  # Store test spans with corrections
        
        print(f"\n{'='*80}")
        print(f"Processing TEST TSV file: {tsv_file}")
        if xml_file:
            print(f"XML file: {xml_file}")
        print(f"Output file: {output_file}")
        if output_spans_file:
            print(f"Spans output file: {output_spans_file}")
        print(f"{'='*80}\n")
        
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, 1):
                # Ensure sequence_id is treated as string
                sequence_id = str(row['Sequence_ID']).strip()
                question_id = str(row['Question_ID']).strip()
                span_type = str(row['Span_Type']).strip()
                span_start = int(row['Span_Start'])
                span_end = int(row['Span_End'])
                original_span = str(row['Original_Span']).strip()
                
                if self.verbose:
                    print(f"\n[{row_num}] Processing Sequence_ID: {sequence_id}, Question: {question_id}, Type: {span_type}")
                
                search_source = self.determine_search_source(span_type)
                
                # Extract full span from XML if available
                full_span = None
                extraction_successful = False
                
                if xml_file:
                    full_span = self.extract_full_span_from_xml(xml_file, question_id, span_start, span_end)
                    if full_span:
                        extraction_successful = True
                        if self.verbose:
                            print(f"  Extracted span from XML: {full_span[:50]}...")
                    else:
                        full_span = original_span
                        if self.verbose:
                            print(f"  Using original span: {original_span[:50]}...")
                else:
                    full_span = original_span
                    if self.verbose:
                        print(f"  Using original span: {original_span[:50]}...")
                
                # Match the span using the matcher
                match_result = self.matcher.match_span_with_verse_splitting(full_span, search_source, verbose=False)
                
                # Determine correction for submission
                if match_result == "خطأ":
                    correction = "خطأ"
                    if self.verbose:
                        print(f"  Result: No match found")
                else:
                    if isinstance(match_result, dict) and 'text' in match_result:
                        correction = str(match_result['text']).strip()  # Ensure string
                        confidence = match_result.get('confidence', 0.0)
                        if self.verbose:
                            print(f"  Result: Found match (confidence: {confidence:.3f})")
                            print(f"    → {correction[:100]}...")
                    else:
                        correction = "خطأ"
                        if self.verbose:
                            print(f"  Result: Invalid match result")
                
                # Add to submission data with explicit string conversion
                submission_data.append({
                    'Sequence_ID': sequence_id,  # Already converted to string above
                    'Correction': correction     # Already converted to string above
                })
                
                # Store test span with correction data
                test_span_data = {
                    'row_number': row_num,
                    'Sequence_ID': sequence_id,
                    'Question_ID': question_id,
                    'Span_Type': span_type,
                    'Span_Start': span_start,
                    'Span_End': span_end,
                    'Original_Span': original_span,
                    'Extracted_Span': full_span if extraction_successful else None,
                    'Extraction_Source': 'XML' if extraction_successful else 'TSV_Original',
                    'Final_Span_Used': full_span,  # The span actually used for matching
                    'System_Correction': correction,
                    'Confidence': round(match_result['confidence'], 3) if match_result != "خطأ" else 0.0,
                    'Match_Details': {
                        'source_type': search_source,
                        'threshold_used': match_result.get('threshold_used', 0.0) if match_result != "خطأ" else 0.0,
                        'match_type': match_result.get('match_type', 'no_match') if match_result != "خطأ" else 'no_match',
                        'is_cross_ayah': match_result.get('is_cross_ayah', False) if match_result != "خطأ" else False,
                        'num_ayahs_combined': match_result.get('num_ayahs_combined', 0) if match_result != "خطأ" else 0
                    } if match_result != "خطأ" else None,
                    'Timestamps': {
                        'processed_at': datetime.now().isoformat(),
                        'extraction_successful': extraction_successful
                    }
                }
                
                # Add source-specific details to test span data
                if match_result != "خطأ" and isinstance(match_result, dict):
                    if match_result.get('source') == 'quran':
                        test_span_data['Quran_Details'] = {
                            'surah_id': match_result.get('surah_id'),
                            'surah_name': match_result.get('surah_name'),
                            'ayah_id': match_result.get('ayah_id'),
                            'ayah_ids': match_result.get('ayah_ids', [match_result.get('ayah_id')]),
                            'is_cross_ayah': match_result.get('is_cross_ayah', False)
                        }
                    elif match_result.get('source') == 'hadith':
                        test_span_data['Hadith_Details'] = {
                            'hadithID': match_result.get('hadithID'),
                            'BookID': match_result.get('BookID'),
                            'title': match_result.get('title')
                        }
                
                test_spans_with_corrections.append(test_span_data)
        
        # Save submission TSV file
        self.save_submission_tsv(submission_data, output_file)
        if self.verbose:
            print(f"\nSubmission file saved: {output_file}")
        
        # Save test spans with corrections
        if output_spans_file:
            self.save_json(test_spans_with_corrections, output_spans_file)
            if self.verbose:
                print(f"Test spans with corrections saved to: {output_spans_file}")
        
        if self.verbose:
            print(f"Total sequences processed: {len(submission_data)}")
        
        return submission_data, test_spans_with_corrections


def main():
    """Main function for processing test set with re-ranking"""
    
    print("\n" + "="*80)
    print(" QURAN AND HADITH TEST SET PROCESSOR WITH RE-RANKING")
    print(" For Competition Submission Format with Enhanced Re-ranking")
    print("="*80)
    
    # Configuration for test set
    config = {
            'test_tsv_file': "../datasets/Test_Subtask_1C_USER.tsv",
            'test_xml_file': "../datasets/Test_Subtask_1C.xml",
            'output_tsv': "test_q8_h7.tsv",
            'output_test_spans': "test_q8_h7.json",
            'quran_index_dir': "quran_index",
            'hadith_index_dir': "hadith_index",
            'quran_threshold': 0.8, 
            'hadith_threshold': 0.7,
            'reranker_model': "BAAI/bge-reranker-v2-m3",
            'use_hf_reranker': True,  # Enabled
            'verbose': True 
        }
    
    print("\nConfiguration:")
    print(f"  Test TSV file: {config['test_tsv_file']}")
    print(f"  Test XML file: {config['test_xml_file']}")
    print(f"  Output TSV: {config['output_tsv']}")
    print(f"  Test spans file: {config['output_test_spans']}")
    print(f"  Quran threshold: {config['quran_threshold']:.0%}")
    print(f"  Hadith threshold: {config['hadith_threshold']:.0%}")
    print(f"  HF Re-ranking: {'Enabled' if config['use_hf_reranker'] else 'Disabled'}")
    if config['use_hf_reranker']:
        print(f"  Re-ranker model: {config['reranker_model']}")
    
    try:
        print("\nInitializing matcher...")
        
        # Initialize matcher with re-ranker
        matcher = QuranHadithSpanMatcher(
            quran_index_dir=config['quran_index_dir'],
            hadith_index_dir=config['hadith_index_dir'],
            verbose=config['verbose'],
            use_hf_reranker=config['use_hf_reranker'],
            reranker_model=config['reranker_model']
        )
        
        # Set thresholds based on dev set performance
        matcher.quran_similarity_threshold = config['quran_threshold']
        matcher.hadith_similarity_threshold = config['hadith_threshold']
        
        # Initialize processor
        processor = SpanCorrector(matcher, verbose=config['verbose'])
        
        print("Matcher and processor initialized")
        print(f"  Quran threshold: {matcher.quran_similarity_threshold:.2f}")
        print(f"  Hadith threshold: {matcher.hadith_similarity_threshold:.2f}")
        print(f"  Cross-ayah matching: Enabled for Quran")
        if config['use_hf_reranker']:
            reranker_status = "Available" if matcher.reranker and matcher.reranker.is_available() else "Fallback to rule-based"
            print(f"  HF Re-ranking: {reranker_status}")
        print(f"Span storage: Enabled")
        
        # Process the test TSV file
        print(f"\nProcessing test TSV file...")
        
        submission_data, test_spans_with_corrections = processor.process_test_tsv_file(
            tsv_file=config['test_tsv_file'],
            xml_file=config['test_xml_file'],
            output_file=config['output_tsv'],
            output_spans_file=config['output_test_spans']
        )
        
        # Print submission and spans statistics
        processor.print_submission_stats(submission_data)
        processor.print_spans_stats(test_spans_with_corrections)
        
        
        print(f"\nTest processing completed!")
        print(f"Submission files:")
        print(f"  - TSV file: {config['output_tsv']}")
        print(f"  - Test spans: {config['output_test_spans']}")
        print(f"\nReady for submission!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
