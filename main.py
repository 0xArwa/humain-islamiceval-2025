#!/usr/bin/env python3
"""
Islamic Shared Task 2025 - Unified Main Entry Point

This script provides a unified interface to run all three subtasks:
- Subtask A: Span Detection for Quran and Hadith
- Subtask B: Span Verification for Quran and Hadith 
- Subtask C: Span Correction for Quran and Hadith

Usage:
    python main.py --task A --mode dev
    python main.py --task B --tsv data.tsv --xml data.xml
    python main.py --task C --input test_data.tsv --output corrected.tsv
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess


def run_subtask_a(mode: str):
    """Run Subtask A: Span Detection"""
    print("🔍 Running Subtask A: Span Detection")
    
    subtask_a_dir = Path("subtask_A")
    if not subtask_a_dir.exists():
        print("❌ Error: subtask_A directory not found")
        return False
    
    cmd = [sys.executable, "main.py", "--mode", mode]
    
    try:
        result = subprocess.run(cmd, cwd=subtask_a_dir, check=True)
        print("✅ Subtask A completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Subtask A: {e}")
        return False


def run_subtask_b(tsv_path: str = None, xml_path: str = None, csv_path: str = None, 
                  quran_path: str = None, hadith_path: str = None, 
                  text_types: list = None, workers: int = 4, output: str = None):
    """Run Subtask B: Span Verification"""
    print("✅ Running Subtask B: Span Verification")
    
    subtask_b_dir = Path("subtask_B")
    if not subtask_b_dir.exists():
        print("❌ Error: subtask_B directory not found")
        return False
    
    # Build command
    cmd = [sys.executable, "span_checker.py"]
    
    if tsv_path and xml_path:
        cmd.extend(["--tsv", tsv_path, "--xml", xml_path])
    elif csv_path:
        cmd.extend(["--csv", csv_path])
    else:
        # Use defaults
        cmd.extend(["--tsv", "datasets_B/Test_Subtask_1B/Test_Subtask_1B_USER.tsv"])
        cmd.extend(["--xml", "datasets_B/Test_Subtask_1B/Test_Subtask_1B.xml"])
    
    if quran_path:
        cmd.extend(["--quran", quran_path])
    if hadith_path:
        cmd.extend(["--hadith", hadith_path])
    if text_types:
        cmd.extend(["--types"] + text_types)
    if workers:
        cmd.extend(["--workers", str(workers)])
    if output:
        cmd.extend(["--output", output])
    
    try:
        result = subprocess.run(cmd, cwd=subtask_b_dir, check=True)
        print("✅ Subtask B completed successfully")
        
        # Check for submission file
        submission_file = subtask_b_dir / "submission.tsv"
        if submission_file.exists():
            print(f"📁 Submission file created: {submission_file}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Subtask B: {e}")
        return False


def run_subtask_c(input_file: str = None, output_file: str = None, use_reranker: bool = True):
    """Run Subtask C: Span Correction"""
    print("🔧 Running Subtask C: Span Correction")
    
    subtask_c_dir = Path("subtask_C")
    if not subtask_c_dir.exists():
        print("❌ Error: subtask_C directory not found")
        return False
    
    # Build command for span correction
    cmd = [sys.executable, "span_corrector.py"]
    
    if input_file:
        cmd.extend(["--input", input_file])
    if output_file:
        cmd.extend(["--output", output_file])
    if not use_reranker:
        cmd.append("--no-reranker")
    
    try:
        result = subprocess.run(cmd, cwd=subtask_c_dir, check=True)
        print("✅ Subtask C completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Subtask C: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Islamic Shared Task 2025 - Unified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Subtask A in development mode
    python main.py --task A --mode dev
    
    # Run Subtask A in test mode
    python main.py --task A --mode test
    
    # Run Subtask B with TSV and XML files
    python main.py --task B --tsv annotations.tsv --xml questions.xml
    
    # Run Subtask B with CSV file
    python main.py --task B --csv processed_data.csv
    
    # Run Subtask C for span correction
    python main.py --task C --input test_data.tsv --output corrected.tsv
    
    # Run all tasks sequentially
    python main.py --task ALL --mode dev
        """
    )
    
    parser.add_argument('--task', choices=['A', 'B', 'C', 'ALL'], required=True,
                       help='Which subtask to run (A=Detection, B=Verification, C=Correction, ALL=Run all)')
    
    # Subtask A arguments
    parser.add_argument('--mode', choices=['dev', 'test'], 
                       help='Mode for Subtask A (dev or test)')
    
    # Subtask B arguments
    parser.add_argument('--tsv', help='Path to TSV annotation file for Subtask B')
    parser.add_argument('--xml', help='Path to XML questions file for Subtask B')
    parser.add_argument('--csv', help='Path to processed CSV file for Subtask B')
    parser.add_argument('--quran', default='datasets/quranic_verses.json',
                       help='Path to Quranic verses JSON')
    parser.add_argument('--hadith', default='datasets/six_hadith_books.json',
                       help='Path to Hadith books JSON')
    parser.add_argument('--types', nargs='+', choices=['Ayah', 'Hadith'], 
                       default=['Ayah', 'Hadith'], help='Text types to evaluate')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--output', help='Output file for Subtask B')
    
    # Subtask C arguments
    parser.add_argument('--input', help='Input file for Subtask C')
    parser.add_argument('--no-reranker', action='store_true', help='Disable reranker for Subtask C')
    
    args = parser.parse_args()
    
    print("🕌 Islamic Shared Task 2025 - Unified Interface")
    print("=" * 50)
    
    success = True
    
    if args.task == 'A' or args.task == 'ALL':
        if not args.mode:
            print("❌ Error: --mode is required for Subtask A")
            return False
        success &= run_subtask_a(args.mode)
    
    if args.task == 'B' or args.task == 'ALL':
        success &= run_subtask_b(
            tsv_path=args.tsv,
            xml_path=args.xml,
            csv_path=args.csv,
            quran_path=args.quran,
            hadith_path=args.hadith,
            text_types=args.types,
            workers=args.workers,
            output=args.output
        )
    
    if args.task == 'C' or args.task == 'ALL':
        success &= run_subtask_c(
            input_file=args.input,
            output_file=args.output if args.task == 'C' else None,
            use_reranker=not args.no_reranker
        )
    
    if success:
        print("\n🎉 All selected tasks completed successfully!")
        return True
    else:
        print("\n❌ Some tasks failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)