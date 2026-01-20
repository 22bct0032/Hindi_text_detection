"""
Inference Script - Run Hindi text detection on images
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from text_detector import HindiTextDetector
import config

class HindiTextInference:
    def __init__(self):
        self.detector = HindiTextDetector()
    
    def process_single_image(self, image_path, output_path=None, show_text=False):
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            show_text: Whether to show detected text on image
        """
        print(f"\nProcessing: {image_path}")
        
        # Detect words
        image, detections = self.detector.detect_words(image_path)
        
        # Get statistics
        stats = self.detector.get_detection_stats(detections)
        print(f"Detected {stats['total_words']} Hindi words")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        
        # Draw bounding boxes
        annotated = self.detector.draw_bounding_boxes(image, detections, show_text)
        
        # Save output
        if output_path is None:
            output_path = os.path.join(
                config.OUTPUT_PATH,
                f"detected_{Path(image_path).name}"
            )
        
        cv2.imwrite(output_path, annotated)
        print(f"Saved output to: {output_path}")
        
        # Save detection details
        self._save_detections(image_path, detections, output_path)
        
        return annotated, detections, stats
    
    def process_batch(self, input_dir, output_dir=None, show_text=False):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            show_text: Whether to show detected text on images
        """
        if output_dir is None:
            output_dir = os.path.join(config.OUTPUT_PATH, "batch_results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
            image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
        
        print(f"\nFound {len(image_files)} images to process")
        
        # Process each image
        all_stats = []
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                output_path = os.path.join(output_dir, f"detected_{img_path.name}")
                _, detections, stats = self.process_single_image(
                    str(img_path),
                    output_path,
                    show_text
                )
                stats['filename'] = img_path.name
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Save summary
        self._save_batch_summary(all_stats, output_dir)
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to: {output_dir}")
    
    def _save_detections(self, image_path, detections, output_path):
        """Save detection details to JSON file"""
        json_path = output_path.replace(f'.{config.SAVE_FORMAT}', '_detections.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_detections = []
        for det in detections:
            serializable_detections.append({
                'bbox': det['bbox'].tolist(),
                'text': det['text'],
                'confidence': float(det['confidence']),
                'width': int(det['width']),
                'height': int(det['height'])
            })
        
        data = {
            'source_image': str(image_path),
            'total_detections': len(detections),
            'detections': serializable_detections
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_batch_summary(self, all_stats, output_dir):
        """Save batch processing summary"""
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        
        # Calculate overall statistics
        total_words = sum(s['total_words'] for s in all_stats)
        avg_words_per_image = total_words / len(all_stats) if all_stats else 0
        
        summary = {
            'total_images_processed': len(all_stats),
            'total_words_detected': total_words,
            'avg_words_per_image': avg_words_per_image,
            'image_stats': all_stats
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nBatch Summary:")
        print(f"  Total images: {len(all_stats)}")
        print(f"  Total words detected: {total_words}")
        print(f"  Average words per image: {avg_words_per_image:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Hindi Handwritten Text Detection')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, help='Output path/directory')
    parser.add_argument('--show_text', action='store_true', help='Show detected text on image')
    
    args = parser.parse_args()
    
    inference = HindiTextInference()
    
    if args.image:
        # Process single image
        inference.process_single_image(args.image, args.output, args.show_text)
    elif args.input_dir:
        # Process batch
        inference.process_batch(args.input_dir, args.output, args.show_text)
    else:
        print("Please provide either --image or --input_dir")
        parser.print_help()

if __name__ == "__main__":
    main()
