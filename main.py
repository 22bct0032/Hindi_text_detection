"""
Main Entry Point for Hindi Handwritten Text Detection System
"""

import argparse
import os
from dataset_analyzer import DatasetAnalyzer
from inference import HindiTextInference
import config

def analyze_dataset():
    """Analyze the dataset"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS MODE")
    print("="*60)
    
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze()
    
def detect_text(args):
    """Run text detection"""
    print("\n" + "="*60)
    print("TEXT DETECTION MODE")
    print("="*60)
    
    inference = HindiTextInference()
    
    if args.image:
        # Single image mode
        inference.process_single_image(
            args.image,
            args.output,
            args.show_text
        )
    elif args.input_dir:
        # Batch mode
        inference.process_batch(
            args.input_dir,
            args.output,
            args.show_text
        )
    else:
        # Default: process entire dataset
        print(f"Processing entire dataset from: {config.DATASET_PATH}")
        inference.process_batch(
            config.DATASET_PATH,
            args.output,
            args.show_text
        )

def main():
    parser = argparse.ArgumentParser(
        description='Hindi Handwritten Text Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze dataset
  python main.py --mode analyze
  
  # Detect text in single image
  python main.py --mode detect --image "DataSet/Hindi Book exercise_page-0001.jpg"
  
  # Process entire dataset
  python main.py --mode detect
  
  # Process specific directory
  python main.py --mode detect --input_dir "DataSet" --output "my_results"
  
  # Show detected text on images
  python main.py --mode detect --show_text
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['analyze', 'detect'],
        default='detect',
        help='Operation mode: analyze dataset or detect text'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for detection'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing images for batch processing'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output path/directory for results'
    )
    
    parser.add_argument(
        '--show_text',
        action='store_true',
        help='Display detected text and confidence on output images'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'analyze':
        analyze_dataset()
    elif args.mode == 'detect':
        detect_text(args)
    
    print("\n" + "="*60)
    print("OPERATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
