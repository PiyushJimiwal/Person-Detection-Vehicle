"""
Configuration file for Person Detection System
Modify these parameters according to your needs
"""

# Model Configuration
MODEL_PATH = 'yolov8n.pt'  # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large)
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence score (0-1)

# Tracking Configuration
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for tracking same person
TRACKING_TIMEOUT = 2.0  # Seconds before removing untracked person

# Display Configuration
SHOW_CONFIDENCE = True  # Show confidence scores on labels
SHOW_FPS = True  # Show FPS counter
SHOW_TOTAL_COUNT = True  # Show total person count

# Video Configuration
DEFAULT_VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
SAVE_OUTPUT = False  # Whether to save output video
OUTPUT_PATH = 'output_detection.mp4'  # Path for output video

# Colors for bounding boxes (BGR format)
BOX_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 128, 0),  # Orange
    (128, 255, 0),  # Lime
]

# Text Configuration
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2
