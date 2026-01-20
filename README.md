# Hindi Handwritten Text Detection System

A comprehensive system for detecting Hindi handwritten text at **word-level granularity** and drawing black bounding boxes around detected words.

## Features

- ✅ **Word-level detection** of Hindi handwritten text
- ✅ **Black bounding boxes** around detected words
- ✅ **EasyOCR-based** detection optimized for Devanagari script
- ✅ **Batch processing** for multiple images
- ✅ **Detailed JSON output** with detection coordinates and confidence scores
- ✅ **Dataset analysis** tools
- ✅ **Configurable parameters** for fine-tuning

## Dataset

The system is trained/tested on 205 Hindi handwritten notebook page images containing:
- Hindi Book exercises (17 pages)
- Hindi work by Rudransh (28 pages)
- Devansh's Hindi work (67 pages)
- Dipika's Hindi work (47 pages)
- Hindi total work (46 pages)

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone or navigate to project directory:**
```bash
cd "c:\Users\ashis\Desktop\project 2"
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** First-time installation of EasyOCR will download Hindi language models (~100MB).

## Usage

### 1. Analyze Dataset

Analyze the dataset to get statistics and sample visualizations:

```bash
python main.py --mode analyze
```

This will:
- Count total images
- Calculate image dimensions and file sizes
- Generate sample image grid
- Save report to `outputs/dataset_samples.png`

### 2. Detect Text in Single Image

Process a single image:

```bash
python main.py --mode detect --image "DataSet/Hindi Book exercise_page-0001.jpg"
```

With text labels:
```bash
python main.py --mode detect --image "DataSet/Hindi Book exercise_page-0001.jpg" --show_text
```

### 3. Batch Process Entire Dataset

Process all images in the dataset:

```bash
python main.py --mode detect
```

Process specific directory:
```bash
python main.py --mode detect --input_dir "DataSet" --output "my_results"
```

### 4. Using Inference Script Directly

For more control, use the inference script:

```bash
# Single image
python inference.py --image "DataSet/Hindi Book exercise_page-0001.jpg" --output "result.jpg"

# Batch processing
python inference.py --input_dir "DataSet" --output "results" --show_text
```

## Output

### Generated Files

For each processed image, the system generates:

1. **Annotated Image** (`detected_*.jpg`): Original image with black bounding boxes
2. **Detection JSON** (`*_detections.json`): Detailed detection data including:
   - Bounding box coordinates
   - Detected text
   - Confidence scores
   - Word dimensions

3. **Batch Summary** (`batch_summary.json`): Overall statistics for batch processing

### Output Structure

```
outputs/
├── detected_Hindi_Book_exercise_page-0001.jpg
├── detected_Hindi_Book_exercise_page-0001_detections.json
├── batch_results/
│   ├── detected_*.jpg
│   ├── *_detections.json
│   └── batch_summary.json
└── dataset_samples.png
```

### Sample JSON Output

```json
{
  "source_image": "DataSet/Hindi Book exercise_page-0001.jpg",
  "total_detections": 45,
  "detections": [
    {
      "bbox": [[120, 85], [180, 85], [180, 110], [120, 110]],
      "text": "गृह",
      "confidence": 0.92,
      "width": 60,
      "height": 25
    }
  ]
}
```

## Configuration

Edit `config.py` to customize:

### Detection Parameters
```python
DETECTION_THRESHOLD = 0.4  # Lower = more detections
TEXT_THRESHOLD = 0.5       # Text recognition confidence
MIN_WORD_WIDTH = 15        # Minimum word width in pixels
MIN_WORD_HEIGHT = 15       # Minimum word height in pixels
```

### Bounding Box Style
```python
BBOX_COLOR = (0, 0, 0)     # Black color (BGR)
BBOX_THICKNESS = 2         # Border thickness
```

### Image Preprocessing
```python
APPLY_DENOISING = True     # Enable/disable denoising
DENOISE_STRENGTH = 10      # Denoising strength (1-20)
```

## Project Structure

```
project 2/
├── config.py              # Configuration settings
├── dataset_analyzer.py    # Dataset analysis tools
├── text_detector.py       # Core detection engine
├── inference.py           # Inference script
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── DataSet/             # Input images (205 files)
├── outputs/             # Detection results
└── models/              # Model checkpoints (if any)
```

## How It Works

1. **Preprocessing**: Images are denoised and contrast-enhanced using CLAHE
2. **Detection**: EasyOCR detects text regions at word-level
3. **Filtering**: Detections are filtered for:
   - Hindi/Devanagari characters
   - Size constraints (min/max width/height)
   - Confidence threshold
4. **Visualization**: Black bounding boxes are drawn around valid detections
5. **Output**: Annotated images and JSON data are saved

## Technical Details

### Model
- **Base Model**: EasyOCR with Hindi language support
- **Detection Method**: CRAFT (Character Region Awareness for Text detection)
- **Recognition**: Pre-trained on Hindi/Devanagari script

### Performance
- **Processing Speed**: ~2-5 seconds per image (GPU) / ~10-20 seconds (CPU)
- **Detection Accuracy**: Optimized for handwritten Hindi text
- **Word-level Granularity**: Each word gets individual bounding box

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Solution: Set `GPU_ENABLED = False` in `config.py`

2. **No detections found**
   - Solution: Lower `DETECTION_THRESHOLD` in `config.py`
   - Try: `DETECTION_THRESHOLD = 0.2`

3. **Too many false positives**
   - Solution: Increase thresholds or adjust size constraints
   - Try: `MIN_WORD_WIDTH = 20`, `MIN_WORD_HEIGHT = 20`

4. **EasyOCR installation issues**
   - Solution: Install PyTorch first: `pip install torch torchvision`

## Examples

### Example 1: Quick Test
```bash
# Test on first image
python main.py --mode detect --image "DataSet/Hindi Book exercise_page-0001.jpg"
```

### Example 2: Full Dataset Processing
```bash
# Process all 205 images
python main.py --mode detect --output "full_results"
```

### Example 3: Analysis + Detection
```bash
# First analyze
python main.py --mode analyze

# Then detect
python main.py --mode detect
```

## Future Enhancements

- [ ] Custom model training for better accuracy
- [ ] Line-level and paragraph-level detection modes
- [ ] Real-time detection via webcam
- [ ] GUI interface for easier usage
- [ ] Support for other Indian scripts

## License

This project is for educational purposes.

## Author

Created for Hindi handwritten text detection project.

## Acknowledgments

- EasyOCR for Hindi text detection capabilities
- OpenCV for image processing
- Dataset contributors: Rudransh, Devansh, Dipika
