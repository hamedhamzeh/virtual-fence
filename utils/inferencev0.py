import cv2
from ultralytics import YOLO

def run_simple_inference(
    model_path,
    input_video_path,
    output_video_path,
    conf_threshold=0.25,
    iou_threshold=0.45
):
    """
    Simple YOLOv8 inference on a video file.
    Reads input video, detects objects, draws bounding boxes, and saves to output video.
    """
    
    # 1. Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # 2. Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties for writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 3. Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Processing video: {width}x{height} @ {fps} FPS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 4. Run Inference
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # 5. Draw Detections        
        det_count = 0
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Draw simple rectangle (Green, thickness 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label (Class index or name)
                    # cls_id = int(box.cls[0])
                    # label = model.names[cls_id]
                    # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                det_count += 1

        # Write frame to output
        out.write(frame)
        
        # Progress log every 30 frames
        if frame_count % 30 == 0:
            percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Frame {frame_count}/{total_frames} ({percent:.1f}%) - Detections: {det_count}")

    # Cleanup
    cap.release()
    out.release()
    print(f"Done! Output saved to: {output_video_path}")

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = "../trained_models/BestN.pt"
    INPUT_VIDEO = "../input.mp4"
    OUTPUT_VIDEO = "output_simple_inference.mp4"
    
    CONF_THRESHOLD = 0.2  # Confidence threshold to filter weak detections
    IOU_THRESHOLD = 0.4   # NMS IOU threshold
    # ---------------------

    run_simple_inference(
        model_path=MODEL_PATH,
        input_video_path=INPUT_VIDEO,
        output_video_path=OUTPUT_VIDEO,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
