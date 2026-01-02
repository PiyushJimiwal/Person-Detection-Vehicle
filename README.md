# Person Detection and Counting System for Vehicle Security

A real-time person detection and counting system using YOLOv8 and OpenCV for vehicle security applications. This system detects persons in video feeds (live camera or uploaded video), tracks them with unique IDs, displays bounding boxes with confidence scores, and counts the total number of persons present.

## Features

- ✅ Real-time person detection using YOLOv8
- ✅ Person tracking with unique IDs (Person 1, Person 2, etc.)
- ✅ Confidence score display for each detection
- ✅ Bounding boxes with color-coded labels
- ✅ Total person count display
- ✅ Supports live camera feed and video files
- ✅ Optional video output saving
- ✅ FPS counter for performance monitoring
- ✅ Configurable detection parameters

## Demo

The system displays:
- **Bounding boxes** around each detected person
- **Person ID** (Person 1, Person 2, etc.)
- **Confidence score** (e.g., 0.95 = 95% confidence)
- **Total count** of persons in the frame
- **Real-time FPS** counter

## Requirements

- Python 3.8 or higher
- Webcam (for live detection) or video file
- Windows/Linux/Mac OS

## Installation

### Step 1: Clone or Download the Project

Save all project files to a folder (e.g., `person_detection_vehicle`)

### Step 2: Create a Virtual Environment (Recommended)

```powershell
# Navigate to project directory
cd C:\person_detection_vehicle

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- `opencv-python` - Computer vision library
- `ultralytics` - YOLOv8 implementation
- `numpy` - Numerical computing
- `torch` & `torchvision` - PyTorch (required by YOLO)

### Step 4: Download YOLO Model

The first time you run the script, it will automatically download the YOLOv8 model. You can choose different model sizes:

- `yolov8n.pt` - Nano (fastest, least accurate) ~6MB
- `yolov8s.pt` - Small ~22MB
- `yolov8m.pt` - Medium (default) ~52MB
- `yolov8l.pt` - Large ~87MB
- `yolov8x.pt` - Extra Large (most accurate, slowest) ~136MB

## Usage

### Option 1: Run with Webcam

```powershell
python person_detector.py --source 0
```

### Option 2: Run with Video File

```powershell
python person_detector.py --source "path/to/your/video.mp4"
```

Example:
```powershell
python person_detector.py --source "C:\Videos\traffic_video.mp4"
```

### Option 3: Run with Custom Settings

```powershell
# Use medium model with 60% confidence threshold
python person_detector.py --source 0 --model yolov8m.pt --confidence 0.6

# Save output video
python person_detector.py --source "input.mp4" --output "output.mp4"

# Run without display (useful for saving video only)
python person_detector.py --source "input.mp4" --output "output.mp4" --no-display
```

### Command-Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--source` | Video source (0 for webcam or video path) | `0` | `--source "video.mp4"` |
| `--model` | YOLO model to use | `yolov8n.pt` | `--model yolov8m.pt` |
| `--confidence` | Confidence threshold (0-1) | `0.5` | `--confidence 0.7` |
| `--output` | Path to save output video | `None` | `--output "result.mp4"` |
| `--no-display` | Disable video display window | `False` | `--no-display` |

## Configuration

Edit `config.py` to customize default settings:

```python
# Model settings
MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.5

# Tracking settings
IOU_THRESHOLD = 0.3
TRACKING_TIMEOUT = 2.0

# Display settings
SHOW_CONFIDENCE = True
SHOW_FPS = True
```

## Controls

While the video is playing:
- Press **'q'** to quit the application

## How It Works

1. **Detection**: YOLOv8 detects persons in each video frame
2. **Tracking**: Simple IoU-based tracking assigns consistent IDs to persons across frames
3. **Visualization**: Draws bounding boxes with labels showing:
   - Person ID (Person 1, Person 2, etc.)
   - Confidence score (0.00 to 1.00)
4. **Counting**: Displays total count of detected persons in current frame

## Performance Tips

### For Better Speed:
- Use smaller models: `yolov8n.pt` or `yolov8s.pt`
- Lower the video resolution
- Increase confidence threshold to reduce false positives

### For Better Accuracy:
- Use larger models: `yolov8m.pt`, `yolov8l.pt`, or `yolov8x.pt`
- Lower confidence threshold (e.g., 0.3-0.4)
- Use higher quality video input

## Troubleshooting

### Issue: "Could not open video source"
- **Camera**: Make sure no other application is using the webcam
- **Video file**: Check that the file path is correct and file exists

### Issue: "No module named 'ultralytics'"
```powershell
pip install ultralytics
```

### Issue: Slow performance
- Use a smaller YOLO model (`yolov8n.pt`)
- Reduce video resolution
- Use GPU acceleration (requires CUDA-compatible GPU)

### Issue: False detections
- Increase confidence threshold: `--confidence 0.7`

### Issue: Missing detections
- Decrease confidence threshold: `--confidence 0.3`
- Use a larger model: `--model yolov8m.pt`

## Project Structure

```
person_detection_vehicle/
│
├── person_detector.py      # Main detection script
├── config.py               # Configuration file
├── requirements.txt        # Python dependencies
├── README.md              # Documentation (this file)
│
└── (models will be downloaded here automatically)
    ├── yolov8n.pt
    └── ...
```

## Use Cases

- **Vehicle Security**: Monitor persons entering/exiting vehicles
- **Public Transport**: Count passengers in buses, trains
- **Surveillance**: Track people in restricted areas
- **Occupancy Monitoring**: Count persons in rooms or vehicles
- **Safety Compliance**: Ensure vehicle capacity limits

## Technical Details

- **Framework**: YOLOv8 (You Only Look Once v8)
- **Detection Class**: Person (COCO class ID: 0)
- **Tracking Method**: IoU-based simple tracking
- **Video Processing**: OpenCV (cv2)
- **Model Format**: PyTorch (.pt files)

## Future Enhancements

Potential improvements:
- Advanced tracking (DeepSORT, ByteTrack)
- Person re-identification
- Suspicious activity detection
- Alert system for overcrowding
- Database logging
- Web interface
- Multiple camera support

## License

This project uses:
- **YOLOv8** by Ultralytics (AGPL-3.0 License)
- **OpenCV** (Apache 2.0 License)

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review configuration in `config.py`
3. Verify all dependencies are installed

## Example Commands

```powershell
# Basic webcam detection
python person_detector.py

# High accuracy with medium model
python person_detector.py --source 0 --model yolov8m.pt --confidence 0.6

# Process video and save output
python person_detector.py --source "input_video.mp4" --output "detected_output.mp4"

# Background processing (no display)
python person_detector.py --source "video.mp4" --output "result.mp4" --no-display

# Low latency with nano model
python person_detector.py --source 0 --model yolov8n.pt --confidence 0.4
```

## Credits

- **YOLO**: Ultralytics YOLOv8
- **OpenCV**: Open Source Computer Vision Library
- **PyTorch**: Deep Learning Framework

---

**Note**: First run will download the YOLO model (takes a few seconds to minutes depending on internet speed).

Happy Detecting! 🎯👤
