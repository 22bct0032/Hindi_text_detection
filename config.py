"""
Configuration file for Hindi Handwritten Text Detection System
"""

import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "DataSet")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_data")

# Create directories if they don't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# EasyOCR Configuration
LANGUAGES = ['hi', 'en']  # Hindi and English
GPU_ENABLED = True  # Set to False if no GPU available
DETECTION_THRESHOLD = 0.4  # Confidence threshold for text detection
TEXT_THRESHOLD = 0.5  # Threshold for text recognition

# Word-level Detection Parameters
MIN_WORD_WIDTH = 15  # Minimum width of detected word box in pixels
MIN_WORD_HEIGHT = 15  # Minimum height of detected word box in pixels
MAX_WORD_WIDTH = 500  # Maximum width to filter out false detections
MAX_WORD_HEIGHT = 200  # Maximum height to filter out false detections

# Bounding Box Visualization
BBOX_COLOR = (0, 0, 0)  # Black color in BGR
BBOX_THICKNESS = 2  # Border thickness in pixels
SAVE_FORMAT = 'jpg'  # Output image format

# Image Preprocessing
RESIZE_WIDTH = None  # Set to None to keep original size, or specify width
RESIZE_HEIGHT = None  # Set to None to keep original size, or specify height
APPLY_DENOISING = True  # Apply denoising to improve detection
DENOISE_STRENGTH = 10  # Denoising strength (1-20)

# Dataset Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Parameters (for fine-tuning if needed)
BATCH_SIZE = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Logging
VERBOSE = True  # Print detailed logs
SAVE_INTERMEDIATE_RESULTS = True  # Save intermediate processing steps
