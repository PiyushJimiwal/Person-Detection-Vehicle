# Person Detection Vehicle Security System

## Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run with Webcam (Default: yolov8n.pt)
```powershell
python person_detector.py --source 0
```

### 3. Run with Video File
```powershell
python person_detector.py --source "your_video.mp4"
```

### 4. Save Output Video
```powershell
python person_detector.py --source "input.mp4" --output "output.mp4"
```

### 5. Use Better Model for Higher Accuracy
```powershell
python person_detector.py --source 0 --model yolov8m.pt
```

## Key Features
- Person detection with YOLOv8 (Nano model by default)
- Unique person IDs (Person 1, 2, 3...)
- Confidence scores displayed
- Real-time person counting
- Face/head region tracking
- Bounding boxes with color-coded labels

## Controls
- Press **'q'** to quit while running

## Model Options
- `yolov8n.pt` - Fastest (default)
- `yolov8m.pt` - Balanced accuracy/speed
- `yolov8l.pt` - High accuracy

See README.md for full documentation.
