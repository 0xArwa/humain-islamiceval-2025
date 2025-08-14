"""
Evaluation script for Islamic text verification system.

This script can evaluate both Quran and Hadith verification systems:
1. Reads entries from the processed CSV file 
2. Runs each text through appropriate verification system
3. Compares verification results with ground truth labels
4. Provides detailed accuracy metrics and analysis
"""

import pandas as pd
import os
from typing import Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import re

from quran_verification import QuranVerifier
from verse_span_validator import VerseSpanValidator
from multi_verse_detector import is_multi_verse_span
from hadith_verification import HadithVerifier

# Quran verification parameters
QURAN_CONFIG = {
    'use_strict_substring': False,     # Set to True for strict substring matching
    'multiverse_strategy': 'all',      # 'any', 'majority', 'all', 'high'
    'single_verse_threshold': 0.85,    # Similarity threshold for single verses
    'multi_verse_threshold': 0.85,     # Similarity threshold for multi-verse spans
}

# Hadith verification parameters  
HADITH_CONFIG = {
    'use_strict_substring': False,     # Set to True for strict substring matching
    'multiverse_strategy': 'all',      # 'any', 'majority', 'all', 'high'
    'single_hadith_threshold': 0.85,   # Similarity threshold for single hadiths
    'multi_hadith_threshold': 0.85,    # Similarity threshold for multi-hadith spans
}


# Create compiled regex patterns for better performance
# Only include true Hadith separators
_HADITH_MULTI_PATTERNS = [
    re.compile(r'\(\d+\)'),  # Numbers in parentheses (indicating separate numbered Hadiths)
    re.compile(r'\n'),  # Line breaks (separate paragraphs/Hadiths)
]

def is_multi_text_span(text_span: str, text_type: str) -> bool:
    """
    Detection of multi-text spans using various indicators.
    
    Args:
        text_span (str): Input text span
        text_type (str): 'Ayah' or 'Hadith'
        
    Returns:
        bool: True if likely contains multiple texts
    """
    if not text_span or len(text_span.strip()) < 20:
        return False
    
    # Fast heuristics first - check for obvious multi-text indicators
    if '*' in text_span or '\n' in text_span:
        return True
        
    # For Ayah: use detection only if needed
    if text_type == 'Ayah':
        # Quick check first - if no obvious separators, likely single verse
        if len(text_span.split()) < 50:  # Most single verses are < 50 words
            return False
        return is_multi_verse_span(text_span)
    
    # Optimized multi-hadith detection using pre-compiled patterns
    indicator_count = 0
    for pattern in _HADITH_MULTI_PATTERNS:
        if pattern.search(text_span):
            indicator_count += 1
            if indicator_count >= 1:  # Any legitimate separator indicates multi-hadith
                return True
    
    return False


def process_single_text(
    row_data: tuple, 
    quran_verifier: Union[QuranVerifier, None],
    quran_validator: Union[VerseSpanValidator, None],
    hadith_verifier: Union[HadithVerifier, None]
) -> Dict:
    """
    Process a single text (Ayah or Hadith) for verification.

    Args:
        row_data: Tuple containing (index, row) from DataFrame.iterrows()
        quran_verifier: OptimizedQuranVerifier instance or None
        quran_validator: VerseSpanValidator instance or None  
        hadith_verifier: HadithVerifier instance or None

    Returns:
        Dict: Detailed result for this text
    """
    idx, row = row_data
    question_id = row["Question_ID"]
    text_span = row["Verse_Hadith_Span"] 
    text_type = row["Ayah_Hadith"]  # 'Ayah' or 'Hadith'
    ground_truth = row["Correct_Incorrect"]  # 'Correct' or 'Incorrect'

    # Initialize result structure
    detailed_result = {
        "index": idx,
        "question_id": question_id,
        "text_span": text_span,
        "text_type": text_type,
        "ground_truth": ground_truth,
        "predicted_match": False,
        "similarity_score": 0.0,
        "correct_prediction": False,
        "is_ground_truth_correct": ground_truth == "Correct",
        "is_multi_text": False,
        "error": None
    }

    try:
        # Process based on text type
        if text_type == "Ayah":
            detailed_result.update(process_quran_text(
                text_span, quran_verifier, quran_validator
            ))
        elif text_type == "Hadith":
            detailed_result.update(process_hadith_text(
                text_span, hadith_verifier
            ))
        else:
            detailed_result["error"] = f"No verifier available for {text_type}"
            return detailed_result

        # Calculate final correctness
        detailed_result["correct_prediction"] = (
            detailed_result["predicted_match"] == detailed_result["is_ground_truth_correct"]
        )

    except Exception as e:
        detailed_result["error"] = str(e)

    return detailed_result


def process_quran_text(text_span: str, verifier: QuranVerifier, validator: VerseSpanValidator) -> Dict:
    """Process Quran text verification."""
    # Validate and correct verse span
    validation_result = validator.validate_verse_span(text_span)
    correction_result = validator.correct_verse_span(text_span)
    corrected_span = correction_result["corrected_span"]

    # Check if we have multi-verse spans (optimize by checking corrected span first)
    is_multi_verse_corrected = is_multi_text_span(corrected_span, "Ayah")
    # Only check original if corrected is different and we need it
    is_multi_verse_original = (is_multi_verse_corrected if text_span == corrected_span 
                              else is_multi_text_span(text_span, "Ayah"))

    # Use appropriate verification method with Quran-specific parameters
    if QURAN_CONFIG['use_strict_substring']:
        if is_multi_verse_corrected:
            verification_result = verifier.verify_separated_verses_strict_substring(corrected_span)
            similarity = verification_result["best_overall_similarity"]
            predicted_match = verification_result["overall_match"]
        else:
            verification_result = verifier.verify_verse_strict_substring(corrected_span)
            similarity = verification_result["similarity"]
            predicted_match = verification_result["is_match"]
    else:
        if is_multi_verse_corrected:
            verification_result = verifier.verify_separated_verses(corrected_span, threshold=QURAN_CONFIG['multi_verse_threshold'])
            similarity = verification_result["best_overall_similarity"]
            predicted_match = verification_result["overall_match"]
        else:
            verification_result = verifier.verify_verse(corrected_span, threshold=QURAN_CONFIG['single_verse_threshold'])
            similarity = verification_result["similarity"]
            predicted_match = verification_result["is_match"]

    # Extract matched info
    if is_multi_verse_corrected:
        matched_surah = ""
        matched_ayah = 0
        if verification_result.get("verification_results"):
            for verse_result in verification_result["verification_results"]:
                if verse_result["is_match"]:
                    matched_surah = verse_result.get("surah_name", "")
                    matched_ayah = verse_result.get("ayah_id", 0)
                    break
    else:
        matched_surah = verification_result.get("surah_name", "")
        matched_ayah = verification_result.get("ayah_id", 0)

    return {
        "corrected_span": corrected_span,
        "predicted_match": predicted_match,
        "similarity_score": similarity,
        "original_similarity_score": 0.0,  # Could add this
        "similarity_improvement": 0.0,  # Could calculate this
        "matched_surah": matched_surah,
        "matched_ayah": matched_ayah,
        "validation_issues": validation_result["issues"],
        "is_span_valid": validation_result["is_valid"],
        "corrections_applied": correction_result["corrections_applied"],
        "correction_needed": correction_result["correction_needed"],
        "is_multi_text": is_multi_verse_corrected,
    }


def process_hadith_text(text_span: str, verifier: HadithVerifier) -> Dict:
    """Process Hadith text verification."""
    # Check if we have multi-hadith spans
    is_multi_hadith = is_multi_text_span(text_span, "Hadith")

    # Use appropriate verification method with Hadith-specific parameters
    if HADITH_CONFIG['use_strict_substring']:
        if is_multi_hadith:
            # Note: Need to implement verify_separated_hadiths_strict_substring
            verification_result = verifier.verify_separated_hadiths(text_span)
            similarity = verification_result["best_overall_similarity"] 
            predicted_match = verification_result["overall_match"]
        else:
            verification_result = verifier.verify_hadith_strict_substring(text_span)
            similarity = verification_result["similarity"]
            predicted_match = verification_result["is_match"]
    else:
        if is_multi_hadith:
            verification_result = verifier.verify_separated_hadiths(text_span, threshold=HADITH_CONFIG['multi_hadith_threshold'])
            similarity = verification_result["best_overall_similarity"]
            predicted_match = verification_result["overall_match"]
        else:
            verification_result = verifier.verify_hadith(text_span, threshold=HADITH_CONFIG['single_hadith_threshold'])
            similarity = verification_result["similarity"]
            predicted_match = verification_result["is_match"]

    # Extract matched info
    if is_multi_hadith:
        matched_title = ""
        matched_hadith_id = 0
        matched_book_id = 0
        if verification_result.get("verification_results"):
            for hadith_result in verification_result["verification_results"]:
                if hadith_result["is_match"]:
                    matched_title = hadith_result.get("title", "")
                    matched_hadith_id = hadith_result.get("hadith_id", 0)
                    matched_book_id = hadith_result.get("book_id", 0)
                    break
    else:
        matched_title = verification_result.get("title", "")
        matched_hadith_id = verification_result.get("hadith_id", 0)
        matched_book_id = verification_result.get("book_id", 0)

    return {
        "predicted_match": predicted_match,
        "similarity_score": similarity,
        "matched_title": matched_title,
        "matched_hadith_id": matched_hadith_id,
        "matched_book_id": matched_book_id,
        "is_multi_text": is_multi_hadith,
    }


def load_data_from_tsv_and_xml(
    tsv_path: str,
    xml_path: str
) -> pd.DataFrame:
    """
    Load data from TSV annotations and XML questions.
    
    Args:
        tsv_path (str): Path to TSV annotation file
        xml_path (str): Path to XML questions file
        
    Returns:
        pd.DataFrame: Combined dataframe with Question_ID, text_span, text_type, ground_truth
    """
    # Load TSV annotations
    tsv_df = pd.read_csv(tsv_path, sep='\t')
    print(f"Loaded {len(tsv_df)} annotations from {tsv_path}")
    
    # Parse XML file using regex (handles malformed XML)
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Create mapping from Question_ID to response text using regex
    question_responses = {}
    
    # Find all Question blocks
    question_pattern = r'<Question>(.*?)</Question>'
    questions = re.findall(question_pattern, xml_content, re.DOTALL)
    
    for question_content in questions:
        # Extract ID
        id_match = re.search(r'<ID>(.*?)</ID>', question_content, re.DOTALL)
        # Extract Response
        response_match = re.search(r'<Response>(.*?)</Response>', question_content, re.DOTALL)
        
        if id_match and response_match:
            question_id = id_match.group(1).strip()
            response = response_match.group(1).strip()
            question_responses[question_id] = response
    
    print(f"Loaded {len(question_responses)} questions from {xml_path}")
    
    # Process annotations and extract spans
    processed_data = []
    for _, row in tsv_df.iterrows():
        question_id = row['Question_ID']
        span_start = int(row['Span_Start'])
        span_end = int(row['Span_End'])
        original_span = row['Original_Span']
        
        # Handle different TSV formats
        if 'Label' in row:
            # Format 1: Question_ID, Label, Span_Start, Span_End, Original_Span, Correction
            label = row['Label']
            correction = row.get('Correction', '')
            
            # Determine text type from label
            if 'Ayah' in label or 'ayah' in label:
                text_type = 'Ayah'
            elif 'Hadith' in label or 'hadith' in label:
                text_type = 'Hadith'
            else:
                text_type = 'Unknown'
            
            # Determine correctness from label
            if 'Wrong' in label or 'wrong' in label:
                correctness = 'Incorrect'
            else:
                correctness = 'Correct'
        else:
            # Format 2: Sequence_ID, Question_ID, Span_Start, Span_End, Original_Span, Span_Type
            sequence_id = row.get('Sequence_ID', '')
            text_type = row.get('Span_Type', 'Unknown')
            label = f"Test{text_type}"
            correction = ''
            correctness = 'Correct'  # Test data assumes correct for verification
        
        # Get response text
        if question_id not in question_responses:
            print(f"Warning: No response found for {question_id}")
            continue
            
        response_text = question_responses[question_id]
        
        # Extract actual span from response using indices
        try:
            extracted_span = response_text[span_start:span_end].strip()
        except IndexError:
            print(f"Warning: Span indices out of range for {question_id}")
            extracted_span = original_span
        
        processed_data.append({
            'Question_ID': question_id,
            'Verse_Hadith_Span': extracted_span,
            'Ayah_Hadith': text_type,
            'Correct_Incorrect': correctness,
            'Original_Span': original_span,
            'Correction': correction,
            'Span_Start': span_start,
            'Span_End': span_end,
            'Label': label
        })
    
    return pd.DataFrame(processed_data)


def evaluate_islamic_text_verification(
    tsv_path: str = None,
    xml_path: str = None,
    csv_path: str = None,
    quranic_verses_path: str = None,
    hadith_books_path: str = None, 
    text_types: list = None,
    max_workers: int = None
) -> Dict:
    """
    Evaluate Islamic text verification system against labeled data.

    Args:
        csv_path (str): Path to the processed CSV file
        quranic_verses_path (str): Path to quranic_verses.json (optional)
        hadith_books_path (str): Path to six_hadith_books.json (optional)
        text_types (list): Types to evaluate ['Ayah', 'Hadith'] (default: both)
        max_workers (int): Maximum number of worker threads

    Returns:
        Dict: Evaluation results with metrics and detailed analysis
    """
    # Load data from TSV/XML or CSV
    if tsv_path and xml_path:
        df = load_data_from_tsv_and_xml(tsv_path, xml_path)
        print(f"Loaded {len(df)} entries from TSV and XML files")
    elif csv_path:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
    else:
        raise ValueError("Must provide either (tsv_path and xml_path) or csv_path")

    # Filter by text types
    if text_types is None:
        text_types = ['Ayah', 'Hadith']
    
    filtered_df = df[df["Ayah_Hadith"].isin(text_types)].copy()
    print(f"Found {len(filtered_df)} entries to evaluate: {dict(filtered_df['Ayah_Hadith'].value_counts())}")

    # Initialize verifiers
    quran_verifier = None
    quran_validator = None
    hadith_verifier = None

    if 'Ayah' in text_types and quranic_verses_path:
        quran_verifier = QuranVerifier(quranic_verses_path, multiverse_strategy=QURAN_CONFIG['multiverse_strategy'])
        quran_validator = VerseSpanValidator()
        print(f"Initialized Quran verifier with {len(quran_verifier.reference_verses)} verses")

    if 'Hadith' in text_types and hadith_books_path:
        hadith_verifier = HadithVerifier(hadith_books_path, multiverse_strategy=HADITH_CONFIG['multiverse_strategy'])
        print(f"Initialized Hadith verifier with {len(hadith_verifier.reference_hadiths)} hadiths")

    # Evaluation results
    results = {
        "total_texts": len(filtered_df),
        "text_types": text_types,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "detailed_results": [],
        "average_similarity": 0.0,
        "errors": 0,
    }

    # Process entries
    use_parallel = max_workers is not None and max_workers > 1
    start_time = time.time()

    if use_parallel:
        print(f"\n=== Processing {len(filtered_df)} entries with {max_workers} workers (Parallel) ===")

        row_data_list = list(filtered_df.iterrows())
        completed_count = 0
        progress_lock = threading.Lock()

        def update_progress():
            nonlocal completed_count
            with progress_lock:
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(row_data_list):
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    print(f"Progress: {completed_count}/{len(row_data_list)} ({completed_count / len(row_data_list) * 100:.1f}%) - {rate:.1f} texts/sec", flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {
                executor.submit(
                    process_single_text, row_data, quran_verifier, quran_validator, hadith_verifier
                ): row_data
                for row_data in row_data_list
            }

            for future in as_completed(future_to_data):
                try:
                    detailed_result = future.result()
                    results["detailed_results"].append(detailed_result)
                    update_metrics(results, detailed_result)
                    update_progress()
                except Exception as e:
                    print(f"Error processing: {e}")
                    results["errors"] += 1
                    update_progress()

        results["detailed_results"].sort(key=lambda x: x["index"])

    else:
        print(f"\n=== Processing {len(filtered_df)} entries (Sequential) ===")
        
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 and idx > 0 else 0
                print(f"Progress: {idx} ({rate:.1f} texts/sec)", end=".", flush=True)

            detailed_result = process_single_text((idx, row), quran_verifier, quran_validator, hadith_verifier)
            results["detailed_results"].append(detailed_result)
            update_metrics(results, detailed_result)

    elapsed_total = time.time() - start_time
    processing_method = f"{max_workers} workers (parallel)" if use_parallel else "sequential"
    print(f"\nCompleted processing {results['total_texts']} entries using {processing_method} in {elapsed_total:.2f} seconds ({results['total_texts'] / elapsed_total:.1f} texts/sec)")

    # Calculate final metrics
    calculate_final_metrics(results)
    
    return results


def update_metrics(results: Dict, detailed_result: Dict):
    """Update evaluation metrics with a single result."""
    if detailed_result.get("error"):
        results["errors"] += 1
        return

    predicted_correct = detailed_result["predicted_match"]
    is_ground_truth_correct = detailed_result["is_ground_truth_correct"]

    if predicted_correct == is_ground_truth_correct:
        results["correct_predictions"] += 1
        if is_ground_truth_correct:
            results["true_positives"] += 1
        else:
            results["true_negatives"] += 1
    else:
        results["incorrect_predictions"] += 1
        if predicted_correct and not is_ground_truth_correct:
            results["false_positives"] += 1
        elif not predicted_correct and is_ground_truth_correct:
            results["false_negatives"] += 1


def calculate_final_metrics(results: Dict):
    """Calculate final evaluation metrics."""
    # Average similarity
    similarities = [r["similarity_score"] for r in results["detailed_results"] if not r.get("error")]
    results["average_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0

    # Standard metrics
    accuracy = results["correct_predictions"] / (results["total_texts"] - results["errors"]) if results["total_texts"] > results["errors"] else 0
    precision = results["true_positives"] / (results["true_positives"] + results["false_positives"]) if (results["true_positives"] + results["false_positives"]) > 0 else 0
    recall = results["true_positives"] / (results["true_positives"] + results["false_negatives"]) if (results["true_positives"] + results["false_negatives"]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results["metrics"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def print_evaluation_report(results: Dict):
    """Print a comprehensive evaluation report."""
    metrics = results["metrics"]
    
    print("\n" + "=" * 60)
    print("ISLAMIC TEXT VERIFICATION SYSTEM EVALUATION REPORT")
    print("=" * 60)

    print(f"\nOVERALL METRICS:")
    print(f"  Total entries evaluated: {results['total_texts']}")
    print(f"  Text types: {', '.join(results['text_types'])}")
    print(f"  Correct predictions: {results['correct_predictions']}")
    print(f"  Incorrect predictions: {results['incorrect_predictions']}")
    if results["errors"] > 0:
        print(f"  Errors: {results['errors']}")
    print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)")

    print(f"\nDETAILED METRICS:")
    print(f"  True Positives (Correct → Match): {results['true_positives']}")
    print(f"  True Negatives (Incorrect → No Match): {results['true_negatives']}")  
    print(f"  False Positives (Incorrect → Match): {results['false_positives']}")
    print(f"  False Negatives (Correct → No Match): {results['false_negatives']}")

    print(f"\nPERFORMANCE METRICS:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  Average Similarity: {results['average_similarity']:.3f}")

    # Show breakdown by text type
    if len(results['text_types']) > 1:
        print(f"\nBREAKDOWN BY TEXT TYPE:")
        for text_type in results['text_types']:
            type_results = [r for r in results['detailed_results'] if r['text_type'] == text_type and not r.get('error')]
            if type_results:
                # Calculate metrics for this text type
                correct = sum(1 for r in type_results if r['correct_prediction'])
                accuracy = correct / len(type_results)
                avg_sim = sum(r['similarity_score'] for r in type_results) / len(type_results)
                
                # Calculate precision, recall, F1 for this text type
                tp = sum(1 for r in type_results if r['predicted_match'] and r['is_ground_truth_correct'])
                fp = sum(1 for r in type_results if r['predicted_match'] and not r['is_ground_truth_correct'])
                fn = sum(1 for r in type_results if not r['predicted_match'] and r['is_ground_truth_correct'])
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"  {text_type}: {correct}/{len(type_results)} correct ({accuracy:.1%})")
                print(f"    Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
                print(f"    Avg similarity: {avg_sim:.3f}")



def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Islamic text verification system')
    parser.add_argument('--tsv', default=None, help='Path to TSV annotation file')
    parser.add_argument('--xml', default=None, help='Path to XML questions file')
    parser.add_argument('--csv', default='../datasets/subtaskB_processed_data.csv', help='Path to processed CSV file')
    parser.add_argument('--quran', default='../datasets/quranic_verses.json', help='Path to Quranic verses JSON')
    parser.add_argument('--hadith', default='../datasets/six_hadith_books.json', help='Path to Hadith books JSON') 
    parser.add_argument('--types', nargs='+', choices=['Ayah', 'Hadith'], default=['Ayah', 'Hadith'], help='Text types to evaluate')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', default='task_b_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle TSV/XML or CSV input
    tsv_path = os.path.join(base_dir, args.tsv) if args.tsv else None
    xml_path = os.path.join(base_dir, args.xml) if args.xml else None
    csv_path = os.path.join(base_dir, args.csv) if args.csv else None
    
    quran_path = os.path.join(base_dir, args.quran) if 'Ayah' in args.types else None
    hadith_path = os.path.join(base_dir, args.hadith) if 'Hadith' in args.types else None
    
    # Check file existence
    if tsv_path and xml_path:
        if not os.path.exists(tsv_path):
            print(f"Error: TSV file not found: {tsv_path}")
            return
        if not os.path.exists(xml_path):
            print(f"Error: XML file not found: {xml_path}")
            return
    elif csv_path:
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return
    else:
        print("Error: Must provide either --tsv and --xml, or --csv")
        return
        
    if quran_path and not os.path.exists(quran_path):
        print(f"Error: Quranic verses file not found: {quran_path}")
        return
        
    if hadith_path and not os.path.exists(hadith_path):
        print(f"Error: Hadith books file not found: {hadith_path}")
        return

    # Run evaluation
    results = evaluate_islamic_text_verification(
        tsv_path=tsv_path,
        xml_path=xml_path,
        csv_path=csv_path,
        quranic_verses_path=quran_path,
        hadith_books_path=hadith_path,
        text_types=args.types,
        max_workers=args.workers
    )

    # Print report
    print_evaluation_report(results)

    # Save detailed results
    detailed_df = pd.DataFrame(results["detailed_results"])
    output_path = os.path.join(base_dir, args.output)
    detailed_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDetailed results saved to: {output_path}")
    
    # Save submission format (sequence_id + prediction)
    submission_path = os.path.join(base_dir, "submission.tsv")
    with open(submission_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results["detailed_results"], 1):
            if not result.get("error"):
                prediction = "Correct" if result["predicted_match"] else "Incorrect"
                f.write(f"{i}\t{prediction}\n")
    print(f"Submission format saved to: {submission_path}")


if __name__ == "__main__":
    main()