"""
Subtask A
Span Detection for Quran and Hadith

Usage:
    python main.py --mode dev
    python main.py --mode test
"""

import argparse
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from data_processor import DataProcessor

def setup_environment(mode: str) -> Config:
    """Setup environment configuration based on mode"""
    # Set environment variable
    os.environ['ENVIRONMENT'] = mode
    
    # Load configuration
    config = Config.from_env()
    
    # Validate configuration
    if not config.azure_openai_api_key or config.azure_openai_api_key == 'your_api_key_here':
        print("Error: Please set AZURE_OPENAI_API_KEY in .env file")
        sys.exit(1)
    
    # Check if dataset files exist
    dataset_path = config.get_dataset_xml()
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        sys.exit(1)
    
    return config

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="Islamic Shared Task 2025 - Subtask A: Span Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dev     # Process development dataset
  python main.py --mode test    # Process test dataset
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['dev', 'test'],
        required=True,
        help='Processing mode: dev for development dataset, test for test dataset'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing (overrides .env setting)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of worker threads for parallel processing (overrides .env setting)'
    )
    
    args = parser.parse_args()
    
    print(f"{args.mode} mode")
    print("="*60)
    
    try:
        # Setup configuration
        config = setup_environment(args.mode)
        
        # Override config with command line arguments
        if args.parallel:
            config.enable_multiprocessing = True
        
        if args.workers:
            config.max_workers = args.workers
        
        print(f"Configuration:")
        print(f"  Environment: {config.environment}")
        print(f"  Dataset: {config.get_dataset_xml()}")
        print(f"  Model: {config.azure_openai_model}")
        print(f"  Parallel processing: {config.enable_multiprocessing}")
        if config.enable_multiprocessing:
            print(f"  Max workers: {config.max_workers}")
        print()
        
        # Initialize processor and run pipeline
        processor = DataProcessor(config)
        pred_file, results_file = processor.run_full_pipeline()
        
        print("\n" + "="*60)
        print("Processing completed successfully!")
        print(f"  Predictions file: {pred_file}")
        print(f"  Results file: {results_file}")
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()