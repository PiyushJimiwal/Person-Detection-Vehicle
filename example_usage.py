"""
Example usage script for Person Detection System
Demonstrates different ways to use the person detector
"""

from person_detector import PersonDetector

# Example 1: Basic webcam detection
print("Example 1: Basic webcam detection")
print("=" * 50)
detector = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.5)
# detector.process_video(video_source=0)  # Uncomment to run

# Example 2: Process video file
print("\nExample 2: Process video file and save output")
print("=" * 50)
detector2 = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.5)
# detector2.process_video(
#     video_source='input_video.mp4',
#     output_path='output_detected.mp4',
#     display=True
# )  # Uncomment to run

# Example 3: High accuracy detection with larger model
print("\nExample 3: High accuracy detection")
print("=" * 50)
detector3 = PersonDetector(model_path='yolov8m.pt', confidence_threshold=0.6)
# detector3.process_video(video_source=0)  # Uncomment to run

# Example 4: Background processing (no display)
print("\nExample 4: Background processing")
print("=" * 50)
detector4 = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.5)
# detector4.process_video(
#     video_source='input.mp4',
#     output_path='output.mp4',
#     display=False
# )  # Uncomment to run

print("\n" + "=" * 50)
print("To run examples, uncomment the lines in this script")
print("=" * 50)
