# Person Detection Vehicle Security System

## Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run with Webcam
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

## Key Features
- Person detection with YOLO
- Unique person IDs (Person 1, 2, 3...)
- Confidence scores displayed
- Real-time person counting
- Bounding boxes with labels

Press 'q' to quit while running.

See README.md for full documentation.
