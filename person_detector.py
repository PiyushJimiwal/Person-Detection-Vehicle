"""
Person Detection and Counting System for Vehicle Security
Uses YOLOv8 for real-time person detection with confidence scores and tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from collections import defaultdict
import time


class PersonDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, use_fast_mode=False):
        """
        Initialize the Person Detector
        
        Args:
            model_path: Path to YOLO model (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: Minimum confidence score for detection (0-1)
            use_fast_mode: Enable optimizations for faster processing (slightly lower accuracy)
        """
        # Check for GPU availability
        import torch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.use_half = torch.cuda.is_available()  # FP16 only works with CUDA
        
        # Load model with optimizations
        self.model = YOLO(model_path)
        if self.use_half:
            self.model.to(self.device)
        
        self.confidence_threshold = confidence_threshold
        self.person_count = 0
        self.tracked_persons = {}
        self.next_person_id = 1
        self.use_fast_mode = use_fast_mode
        self.frame_skip = 2 if use_fast_mode else 1  # Process every nth frame
        self.frame_count_internal = 0
        self.last_detections = []  # Cache last detections for skipped frames
        
        # Initialize face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Colors for bounding boxes (BGR format)
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 128, 0), (128, 255, 0)
        ]
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for tracking"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def assign_person_id(self, box):
        """Assign ID to detected person using simple tracking"""
        max_iou = 0
        assigned_id = None
        
        # Try to match with existing tracked persons
        for person_id, (prev_box, last_seen) in list(self.tracked_persons.items()):
            iou = self.calculate_iou(box, prev_box)
            if iou > max_iou and iou > 0.3:  # IoU threshold for matching
                max_iou = iou
                assigned_id = person_id
        
        # If no match found, assign new ID
        if assigned_id is None:
            assigned_id = self.next_person_id
            self.next_person_id += 1
        
        # Update tracking
        self.tracked_persons[assigned_id] = (box, time.time())
        
        # Remove old tracked persons (not seen for 1.5 seconds)
        current_time = time.time()
        for person_id, (_, last_seen) in list(self.tracked_persons.items()):
            if current_time - last_seen > 1.5:
                del self.tracked_persons[person_id]
        
        return assigned_id
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            annotated_frame: Frame with bounding boxes and labels
            person_count: Number of persons detected in current frame
        """
        self.frame_count_internal += 1
        annotated_frame = frame.copy()
        
        # Skip frames in fast mode - use cached detections
        if self.use_fast_mode and self.frame_count_internal % self.frame_skip != 0:
            return self._apply_cached_detections(annotated_frame)
        
        # Run YOLO detection with optimized settings
        img_size = 512  # Optimized size for speed/accuracy balance
        
        results = self.model(
            frame, 
            verbose=False, 
            imgsz=img_size, 
            conf=self.confidence_threshold,
            device=self.device,
            half=self.use_half,
            agnostic_nms=True,  # Faster NMS
            max_det=50  # Limit max detections for speed
        )
        
        current_frame_persons = []
        self.last_detections = []  # Reset cache
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # YOLO class ID 0 is 'person'
                if class_id == 0 and confidence >= self.confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract person region for face detection
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Detect face within person region with optimized parameters
                    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Faster face detection settings
                    scale = 1.15
                    neighbors = 4
                    min_size = (25, 25)
                    
                    faces = self.face_cascade.detectMultiScale(
                        gray_roi, 
                        scaleFactor=scale,
                        minNeighbors=neighbors,
                        minSize=min_size,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Use face coordinates if detected, otherwise use upper 30% of person box
                    if len(faces) > 0:
                        # Use the first (largest) detected face
                        fx, fy, fw, fh = faces[0]
                        # Convert face coordinates to frame coordinates
                        face_x1 = x1 + fx
                        face_y1 = y1 + fy
                        face_x2 = face_x1 + fw
                        face_y2 = face_y1 + fh
                    else:
                        # Fallback: use upper 30% of person bounding box (head region)
                        box_height = y2 - y1
                        face_x1 = x1
                        face_y1 = y1
                        face_x2 = x2
                        face_y2 = y1 + int(box_height * 0.3)
                    
                    # Assign person ID
                    person_id = self.assign_person_id((x1, y1, x2, y2))
                    current_frame_persons.append(person_id)
                    
                    # Select color based on person ID
                    color = self.colors[person_id % len(self.colors)]
                    
                    # Cache detection for frame skipping
                    self.last_detections.append({
                        'box': (face_x1, face_y1, face_x2, face_y2),
                        'person_id': person_id,
                        'confidence': confidence,
                        'color': color
                    })
                    
                    # Draw bounding box around face/head region only
                    cv2.rectangle(annotated_frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
                    
                    # Create label with person ID and confidence
                    label = f"Person {person_id}: {confidence:.2f}"
                    
                    # Calculate label size for background
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame,
                        (face_x1, face_y1 - label_height - 10),
                        (face_x1 + label_width, face_y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        label,
                        (face_x1, face_y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        # Update person count
        self.person_count = len(current_frame_persons)
        
        # Display total person count on frame
        count_text = f"Total Persons: {self.person_count}"
        cv2.rectangle(annotated_frame, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            count_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return annotated_frame, self.person_count
    
    def _apply_cached_detections(self, frame):
        """Apply cached detections to skipped frames for speed"""
        annotated_frame = frame.copy()
        
        for detection in self.last_detections:
            face_x1, face_y1, face_x2, face_y2 = detection['box']
            person_id = detection['person_id']
            confidence = detection['confidence']
            color = detection['color']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
            
            # Draw label
            label = f"Person {person_id}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated_frame,
                (face_x1, face_y1 - label_height - 10),
                (face_x1 + label_width, face_y1),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (face_x1, face_y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Display person count
        count_text = f"Total Persons: {len(self.last_detections)}"
        cv2.rectangle(annotated_frame, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            count_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        return annotated_frame, len(self.last_detections)
    
    def process_video(self, video_source, output_path=None, display=True):
        """
        Process video from camera or file
        
        Args:
            video_source: 0 for webcam, or path to video file
            output_path: Path to save output video (optional)
            display: Whether to display video in window
        """
        # Open video source
        if isinstance(video_source, int):
            cap = cv2.VideoCapture(video_source)
            print(f"Opening camera {video_source}...")
        else:
            cap = cv2.VideoCapture(video_source)
            print(f"Opening video file: {video_source}")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
        
        # Initialize video writer if output path is specified
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps, (frame_width, frame_height)
            )
            print(f"Saving output to: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        
        print("Processing video... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("End of video or error reading frame")
                break
            
            frame_count += 1
            
            # Detect persons in frame
            annotated_frame, person_count = self.detect_persons(frame)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(
                annotated_frame,
                fps_text,
                (frame_width - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Write frame to output video
            if video_writer:
                video_writer.write(annotated_frame)
            
            # Display frame
            if display:
                cv2.imshow('Person Detection - Vehicle Security', annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User requested quit")
                    break
        
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Person Detection and Counting for Vehicle Security'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: 0 for webcam, or path to video file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detection (0-1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display window'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Enable fast mode for higher FPS (with frame skipping)'
    )
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (camera index)
    video_source = int(args.source) if args.source.isdigit() else args.source
    
    # Initialize detector
    print(f"Initializing Person Detector with model: {args.model}")
    print(f"Fast mode: {'Enabled' if args.fast else 'Disabled'}")
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_fast_mode=args.fast
    )
    
    # Process video
    detector.process_video(
        video_source=video_source,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
