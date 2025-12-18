"""
YOLOv8 Inference Script with SimpleSORT Tracking
Processes input video and outputs video with bounding boxes and track IDs.
TODO: Includes proper NMS and smoothed tracking.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tracker import SimpleSORT
from visualizer import TrackVisualizer
# from filters import DetectionFilter

from torchvision.ops import nms

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
    # Load YOLOv8 model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

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
    tracker_name = "SimpleSORT tracker" if tracker else "raw detection (no tracking)"
    print(f"Processing frames with {tracker_name}...")
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 inference
        results = model.predict(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False
        )

        detections = []
        confidences = []

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            # Apply proper NMS using torchvision
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            keep_idx = nms(boxes_tensor, scores_tensor, iou_threshold)
            keep_idx = keep_idx.cpu().numpy()

            detections = boxes[keep_idx]
            confidences = scores[keep_idx]

        # Update tracker
        if tracker is not None:
            tracked_objects = tracker.update(np.array(detections), np.array(confidences))
        else:
            # No tracker: just enumerate detections
            tracked_objects = [(i, box, conf, 1) for i, (box, conf) in enumerate(zip(detections, confidences))]

        # Draw tracked boxes
        for track_id, box, confidence, history_len in tracked_objects:
            if detection_filter is None or detection_filter.apply_all_filters(box, history_len):
                TrackVisualizer.draw_tracked_box(
                    frame,
                    box,
                    track_id,
                    confidence,
                    line_width=line_width,
                    show_label=show_labels and tracker is not None
                )

        # Write frame
        out.write(frame)
        frame_count += 1

        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            if tracker is not None:
                stats = tracker.get_statistics()
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | "
                      f"Active tracks: {stats['active_tracks']} | Max ID: {stats['max_track_id']}")
            else:
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) | Detections: {len(detections)}")

    cap.release()
    out.release()

    print(f"\nInference complete! Output saved to {output_video_path}")
    if tracker is not None:
        stats = tracker.get_statistics()
        print(f"Total unique tracks detected: {stats['total_unique_tracks']}")
        print(f"Active tracks in last frame: {stats['active_tracks']}")


if __name__ == "__main__":
    model_path = "trained_models/best-40.pt"
    input_video_path = "input.mp4"
    output_video_path = "output_tracked_simple_sort.mp4"

    # Optionally create a detection filter
    # custom_filter = DetectionFilter(min_box_area=0, min_aspect_ratio=0.5, min_track_duration=1)

    # Initialize SimpleSORT tracker
    tracker = SimpleSORT(max_age=5, min_box_area=1000, max_box_area=5000)

    run_inference_with_tracking(
        model_path=model_path,
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        conf_threshold=0.2,
        iou_threshold=0.5,
        max_det=300,
        show_labels=True,
        line_width=2,
        detection_filter=None,  # or pass custom_filter
        tracker=tracker
    )
