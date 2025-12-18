"""
YOLOv8 Inference Script with Custom Tracking
Processes input video and outputs video with bounding boxes and track IDs.
Uses custom tracking implementation and filtering modules.
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np




def run_inference_with_tracking(
    model_path, 
    input_video_path, 
    output_video_path, 
    conf_threshold=0.25,
    iou_threshold=0.5,
    max_det=1000,
    show_labels=True,
    line_width=2,
    detection_filter=None,
    tracker=None
):
    """
    Run inference on a video using YOLOv8 model with custom tracking.
    
    Args:
        model_path (str): Path to the trained YOLOv8 model
        input_video_path (str): Path to input video
        output_video_path (str): Path to save output video
        conf_threshold (float): Detection confidence threshold
        iou_threshold (float): IOU threshold for NMS
        max_det (int): Maximum number of detections per image
        show_labels (bool): Whether to show track IDs on boxes
        line_width (int): Thickness of bounding box lines
        detection_filter (DetectionFilter, optional): Filter for post-processing. If None, shows all tracks without filtering.
        tracker (CustomTracker, optional): Custom tracker instance. If None, creates new tracker.
    """
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Initialize tracker if not provided
    # if tracker is None:
    #    tracker = CustomTracker(max_frames_missing=30, iou_threshold=0.3)
    
    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {input_video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    tracker_name = "custom tracker" if tracker else "raw detection (no tracking)"
    print(f"Processing frames with {tracker_name}...")
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLOv8 inference (detection only, no built-in tracking)
        results = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False
        )
        
        # Extract detections
        detections = []
        confidences = []
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            detections = boxes
            confidences = confs
        
        # Update tracker with detections
        if tracker is not None:
            tracked_objects = tracker.update(detections, confidences)
        else:
            # No tracking, just raw detections
            # Format: (track_id, box, confidence, history_length)
            tracked_objects = []
            for i, (box, conf) in enumerate(zip(detections, confidences)):
                # Use index as fake ID for color variety in this frame
                tracked_objects.append((i, box, conf, 1))
        
        # Draw filtered tracked objects
        for track_id, box, confidence, history_length in tracked_objects:
            # Apply filters if detection_filter is provided, otherwise show all tracks
            if detection_filter is None or detection_filter.apply_all_filters(box, history_length):
                x1, y1, x2, y2 = box.astype(int)

                np.random.seed(track_id)
                color = tuple(int(c) for c in np.random.randint(0, 255, 3))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_width)

                if show_labels and tracker is not None:
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )

        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            if tracker is not None:
                stats = tracker.get_statistics()
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) | "
                      f"Active tracks: {stats['active_tracks']} | Max ID: {stats['max_track_id']}")
            else:
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) | "
                      f"Detections: {len(detections)}")
    
    cap.release()
    out.release()
    
    # Final statistics
    print(f"\nInference + tracking complete! Saved to {output_video_path}")
    if tracker is not None:
        stats = tracker.get_statistics()
        print(f"Total unique tracks detected: {stats['total_unique_tracks']}")
        print(f"Active tracks in last frame: {stats['active_tracks']}")


if __name__ == "__main__":
    model_path = "trained_models/best-40.pt"
    input_video_path = "input.mp4"
    output_video_path = "output_tracked_custom.mp4"


    # Run inference with custom tracking
    run_inference_with_tracking(
        model_path=model_path,
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        conf_threshold=0.2,  # Lower threshold for crowded scenes
        iou_threshold=0.7,   # Higher NMS threshold to prevent merging close people
        max_det=300,        # Increase for very crowded scenes
        show_labels=True,
        line_width=1,
    )
