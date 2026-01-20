"""
Quick Demo Script - Process a few sample images for demonstration
"""

import os
from pathlib import Path
from inference import HindiTextInference
import config

def run_demo():
    """Run detection on 5 sample images"""
    print("="*60)
    print("HINDI TEXT DETECTION - QUICK DEMO")
    print("="*60)
    
    # Get first 5 images from dataset
    image_files = list(Path(config.DATASET_PATH).glob('*.jpg'))[:5]
    
    print(f"\nProcessing {len(image_files)} sample images...")
    
    # Create output directory
    demo_output = os.path.join(config.OUTPUT_PATH, "demo_results")
    os.makedirs(demo_output, exist_ok=True)
    
    # Initialize detector
    inference = HindiTextInference()
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        output_path = os.path.join(demo_output, f"demo_{i}_{img_path.name}")
        
        try:
            _, detections, stats = inference.process_single_image(
                str(img_path),
                output_path,
                show_text=False
            )
            
            print(f"  ✓ Detected {stats['total_words']} words")
            print(f"  ✓ Avg confidence: {stats['avg_confidence']:.2f}")
            print(f"  ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print(f"Results saved to: {demo_output}")
    print("="*60)

if __name__ == "__main__":
    run_demo()
