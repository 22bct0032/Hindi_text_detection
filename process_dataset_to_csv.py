"""
Batch Process Entire Dataset and Generate CSV Report
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm
from inference import HindiTextInference
import config

def process_dataset_to_csv(dataset_path=config.DATASET_PATH, output_csv='detection_results.csv'):
    """
    Process entire dataset and generate CSV with detection results
    
    Args:
        dataset_path: Path to dataset directory
        output_csv: Output CSV filename
    """
    print("="*60)
    print("BATCH PROCESSING DATASET TO CSV")
    print("="*60)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(dataset_path).glob(f'*{ext}')))
        image_files.extend(list(Path(dataset_path).glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Initialize detector
    inference = HindiTextInference()
    
    # Prepare CSV data
    csv_data = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Create output path
            output_dir = os.path.join(config.OUTPUT_PATH, "batch_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{img_path.name}")
            
            # Process image
            _, detections, stats = inference.process_single_image(
                str(img_path),
                output_path,
                show_text=False
            )
            
            # Extract detected words in reading order
            detected_words = [d['text'] for d in detections]
            detected_words_str = ' '.join(detected_words)
            word_count = len(detections)
            
            # Add row to CSV
            csv_data.append({
                'image_name': img_path.name,
                'predicted_words': detected_words_str,
                'predicted_count': word_count,
                'actual_words': '',  # To be filled manually
                'actual_count': '',  # To be filled manually
                'difference': '',  # To be calculated manually
                'avg_confidence': f"{stats['avg_confidence']:.3f}"
            })
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            csv_data.append({
                'image_name': img_path.name,
                'predicted_words': 'ERROR',
                'predicted_count': 0,
                'actual_words': '',
                'actual_count': '',
                'difference': '',
                'avg_confidence': 0
            })
    
    # Write CSV file
    csv_path = os.path.join(config.OUTPUT_PATH, output_csv)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'image_name', 
            'predicted_words', 
            'predicted_count',
            'actual_words', 
            'actual_count',
            'difference',
            'avg_confidence'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {len(csv_data)}")
    print(f"CSV file saved to: {csv_path}")
    print(f"Annotated images saved to: {output_dir}")
    print(f"\nNote: 'actual_words', 'actual_count', and 'difference' columns")
    print(f"are empty and should be filled manually for evaluation.")
    print(f"{'='*60}\n")
    
    return csv_path

if __name__ == "__main__":
    process_dataset_to_csv()
