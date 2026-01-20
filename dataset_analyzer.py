"""
Dataset Analyzer - Analyzes the Hindi handwritten text dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import config

class DatasetAnalyzer:
    def __init__(self, dataset_path=config.DATASET_PATH):
        self.dataset_path = dataset_path
        self.image_files = self._get_image_files()
        
    def _get_image_files(self):
        """Get all image files from dataset directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(self.dataset_path).glob(f'*{ext}')))
            image_files.extend(list(Path(self.dataset_path).glob(f'*{ext.upper()}')))
        
        return sorted(image_files)
    
    def analyze(self):
        """Perform comprehensive dataset analysis"""
        print(f"Analyzing dataset at: {self.dataset_path}")
        print(f"Total images found: {len(self.image_files)}\n")
        
        if len(self.image_files) == 0:
            print("No images found in dataset!")
            return
        
        # Statistics containers
        widths = []
        heights = []
        aspect_ratios = []
        file_sizes = []
        
        print("Processing images...")
        for img_path in tqdm(self.image_files):
            # Read image
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Collect statistics
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h)
            file_sizes.append(os.path.getsize(img_path) / 1024)  # KB
        
        # Calculate statistics
        print("\n" + "="*60)
        print("DATASET ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nTotal Images: {len(self.image_files)}")
        print(f"Successfully Loaded: {len(widths)}")
        
        print(f"\nImage Dimensions:")
        print(f"  Width  - Min: {min(widths)}px, Max: {max(widths)}px, Avg: {np.mean(widths):.1f}px")
        print(f"  Height - Min: {min(heights)}px, Max: {max(heights)}px, Avg: {np.mean(heights):.1f}px")
        print(f"  Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Avg: {np.mean(aspect_ratios):.2f}")
        
        print(f"\nFile Sizes:")
        print(f"  Min: {min(file_sizes):.1f} KB")
        print(f"  Max: {max(file_sizes):.1f} KB")
        print(f"  Avg: {np.mean(file_sizes):.1f} KB")
        print(f"  Total: {sum(file_sizes)/1024:.1f} MB")
        
        # Display sample images
        self._display_samples()
        
        print("\n" + "="*60)
        print("Analysis complete!")
        print("="*60)
        
        return {
            'total_images': len(self.image_files),
            'widths': widths,
            'heights': heights,
            'aspect_ratios': aspect_ratios,
            'file_sizes': file_sizes
        }
    
    def _display_samples(self, num_samples=6):
        """Display sample images from dataset"""
        print(f"\nDisplaying {num_samples} sample images...")
        
        # Select random samples
        sample_indices = np.random.choice(len(self.image_files), 
                                         min(num_samples, len(self.image_files)), 
                                         replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, img_idx in enumerate(sample_indices):
            img_path = self.image_files[img_idx]
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(img_path.name, fontsize=8)
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(config.OUTPUT_PATH, 'dataset_samples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Sample images saved to: {output_path}")
        plt.close()

if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze()
