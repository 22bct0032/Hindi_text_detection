"""
Hindi Text Detector - Word-level detection using EasyOCR
"""

import cv2
import numpy as np
import easyocr
from typing import List, Tuple
import config

class HindiTextDetector:
    def __init__(self, gpu=config.GPU_ENABLED):
        """Initialize EasyOCR reader for Hindi text detection"""
        print("Initializing EasyOCR reader for Hindi...")
        self.reader = easyocr.Reader(
            config.LANGUAGES,
            gpu=gpu,
            verbose=config.VERBOSE
        )
        print("EasyOCR reader initialized successfully!")
    
    def preprocess_image(self, image):
        """Preprocess image for better detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising if enabled
        if config.APPLY_DENOISING:
            gray = cv2.fastNlMeansDenoising(gray, None, config.DENOISE_STRENGTH, 7, 21)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR for EasyOCR
        preprocessed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return preprocessed
    
    def detect_words(self, image_path: str) -> Tuple[np.ndarray, List[dict]]:
        """
        Detect Hindi words in image
        
        Args:
            image_path: Path to input image
            
        Returns:
            image: Original image
            detections: List of detection dictionaries with bbox, text, confidence
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Detect text with EasyOCR
        results = self.reader.readtext(
            preprocessed,
            detail=1,
            paragraph=False,  # Word-level detection
            min_size=config.MIN_WORD_WIDTH,
            text_threshold=config.TEXT_THRESHOLD,
            low_text=config.DETECTION_THRESHOLD
        )
        
        # Process detections
        detections = []
        for bbox, text, confidence in results:
            # Convert bbox to integer coordinates
            bbox = np.array(bbox, dtype=np.int32)
            
            # Calculate bounding box dimensions
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            width = x_max - x_min
            height = y_max - y_min
            
            # Filter based on size constraints
            if (width < config.MIN_WORD_WIDTH or 
                height < config.MIN_WORD_HEIGHT or
                width > config.MAX_WORD_WIDTH or 
                height > config.MAX_WORD_HEIGHT):
                continue
            
            # Filter for Hindi text (contains Devanagari characters)
            if self._contains_hindi(text):
                detections.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence,
                    'width': width,
                    'height': height
                })
        
        # Sort detections in reading order (top-to-bottom, left-to-right)
        detections = self._sort_reading_order(detections)
        
        return image, detections
    
    def _contains_hindi(self, text: str) -> bool:
        """Check if text contains Hindi/Devanagari characters"""
        # Devanagari Unicode range: U+0900 to U+097F
        for char in text:
            if '\u0900' <= char <= '\u097F':
                return True
        return False
    
    def _sort_reading_order(self, detections):
        """
        Sort detections in reading order: top-to-bottom, left-to-right
        Groups words into lines based on Y-coordinate overlap
        """
        if not detections:
            return detections
        
        # Sort by Y-coordinate (top to bottom) first
        sorted_detections = sorted(detections, key=lambda d: d['bbox'][:, 1].min())
        
        # Group into lines based on Y-coordinate overlap
        lines = []
        current_line = [sorted_detections[0]]
        
        for detection in sorted_detections[1:]:
            current_y = detection['bbox'][:, 1].min()
            last_y = current_line[-1]['bbox'][:, 1].min()
            last_height = current_line[-1]['height']
            
            # If Y-coordinates are close (within half line height), consider same line
            if abs(current_y - last_y) < last_height * 0.5:
                current_line.append(detection)
            else:
                # Sort current line left-to-right and add to lines
                current_line.sort(key=lambda d: d['bbox'][:, 0].min())
                lines.append(current_line)
                current_line = [detection]
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda d: d['bbox'][:, 0].min())
            lines.append(current_line)
        
        # Flatten lines back into single list
        ordered_detections = []
        for line in lines:
            ordered_detections.extend(line)
        
        return ordered_detections
    
    def draw_bounding_boxes(self, image, detections, show_text=False):
        """
        Draw black bounding boxes on detected words
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_text: Whether to display detected text on image
            
        Returns:
            annotated_image: Image with bounding boxes
        """
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            text = detection['text']
            confidence = detection['confidence']
            
            # Draw black border
            cv2.polylines(
                annotated,
                [bbox],
                isClosed=True,
                color=config.BBOX_COLOR,
                thickness=config.BBOX_THICKNESS
            )
            
            # Optionally show text and confidence
            if show_text:
                # Get top-left corner for text placement
                x_min = bbox[:, 0].min()
                y_min = bbox[:, 1].min()
                
                # Draw text background
                label = f"{text} ({confidence:.2f})"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(
                    annotated,
                    (x_min, y_min - label_h - 5),
                    (x_min + label_w, y_min),
                    (255, 255, 255),
                    -1
                )
                
                # Draw text
                cv2.putText(
                    annotated,
                    label,
                    (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1
                )
        
        return annotated
    
    def get_detection_stats(self, detections):
        """Get statistics about detections"""
        if not detections:
            return {
                'total_words': 0,
                'avg_confidence': 0,
                'avg_width': 0,
                'avg_height': 0
            }
        
        confidences = [d['confidence'] for d in detections]
        widths = [d['width'] for d in detections]
        heights = [d['height'] for d in detections]
        
        return {
            'total_words': len(detections),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_width': np.mean(widths),
            'avg_height': np.mean(heights)
        }
