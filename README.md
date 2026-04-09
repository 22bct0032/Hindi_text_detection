# Hindi Handwritten Word Detection System

A robust computer vision system for detecting and localizing individual Hindi (Devanagari) words from handwritten document images using adaptive morphological processing.

## Features

- Word-level detection of Hindi handwritten text with ~96.7% F1-Score
- Angle-robust — works on documents tilted 0° to 70°
- Adaptive processing — dynamic kernel sizes based on text density per band
- Comprehensive filtering — 9-stage post-processing pipeline removes noise
- Batch processing for entire datasets
- No deep learning dependency — pure OpenCV + morphological operations

## Algorithm Overview

The system processes images through a multi-stage pipeline:

1. **Image Acquisition** — Load image, apply inverse perspective correction for tilted documents
2. **Preprocessing** — Grayscale → CLAHE contrast normalization → Background subtraction → Otsu binarization
3. **Word Detection** — Connected Component Analysis → Adaptive band-wise morphological closing
4. **Post-Processing** — Merge → Split → Size/Aspect/Line/Border/Paper/Containment/NMS/Isolation filters
5. **Output Generation** — Reading order sort → Black bounding boxes on original image + JSON data

## Dataset

| Folder | Images | Description |
|--------|--------|-------------|
| `dataSet/` | 111 | Diverse handwritten Hindi document images |

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

```bash
# Clone the repository
git clone https://github.com/22bct0032/Hindi_text_detection.git
cd Hindi_text_detection

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Detection on Dataset

```bash
python run_detection.py
```

This processes the dataset and saves annotated images and JSON files to the `outputs/` directory.

## Project Structure

```
Hindi_text_detection/
├── final_improved.py          # Core detection algorithm
│                              #   - CLAHE contrast normalization
│                              #   - Background subtraction & binarization
│                              #   - Adaptive band-wise morphological closing
│                              #   - 9-stage post-processing filters
│                              #   - Reading order sort & visualization
│
├── run_detection.py           # Batch processor for the dataset
│                              #   - Calls final_improved.py for each image
│                              #   - Saves annotated images + JSON data
│
├── config.py                  # Configuration parameters
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── dataSet/                   # 111 handwritten Hindi images
│
└── outputs/
    ├── dataSet_Output/        # Detection results (annotated images)
    └── evaluation/            # Evaluation graphs & metrics
```

## Key Results

| Metric | Our Algorithm |
|--------|:------------:|
| **F1-Score** | **96.7%** |
| **Precision** | **97.1%** |
| **Recall** | **96.3%** |

## Configuration

Edit `config.py` to customize detection parameters:

```python
BBOX_COLOR = (0, 0, 0)     # Black bounding boxes
BBOX_THICKNESS = 1          # Thin borders
```

## Technical Details

- **Core Engine**: Custom morphological pipeline (no deep learning)
- **Binarization**: Otsu's method with adaptive fallback
- **Morphological Closing**: Dynamic per-band kernel sizing
- **Line Detection**: Center-based vertical grouping
- **Post-Processing**: 9 sequential filters (size, aspect ratio, line, border, paper mask, relative size, containment, NMS, isolation)
- **Libraries**: OpenCV, NumPy, Matplotlib

## License

This project is for educational and research purposes.

## Author

Hindi Handwritten Word Detection — Research Project
