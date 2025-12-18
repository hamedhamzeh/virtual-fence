"""
YOLOv8 Image Inference Script
Processes input image and outputs image with bounding boxes around detected people.
"""

import cv2
from ultralytics import YOLO
import os


def inference_image(model_path, input_image_path, output_image_path, conf_threshold=0.25):
    """
    Run inference on an image using a trained YOLOv8 model.
    
    Args:
        model_path (str): Path to the trained model file (best.pt)
        input_image_path (str): Path to the input image file
        output_image_path (str): Path to save the output image
        conf_threshold (float): Confidence threshold for detections (default: 0.25)
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Read the input image
    print(f"Reading input image: {input_image_path}")
    image = cv2.imread(input_image_path)
    
    if image is None:
        raise ValueError(f"Error: Could not read image file {input_image_path}")
    
    # Run inference on the image
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Draw bounding boxes on the image
    annotated_image = results[0].plot()
    
    # Save the output image
    cv2.imwrite(output_image_path, annotated_image)
    
    # Print detection info
    num_detections = len(results[0].boxes)
    print(f"\nInference complete! Output image saved to: {output_image_path}")
    print(f"Total detections: {num_detections}")


if __name__ == "__main__":
    # File paths
    model_path = "best.pt"
    input_image_path = "Datasets/Mot17/test/MOT17-12/img1/000002.jpg"  # Change this to your input image path
    output_image_path = "output_with_boxes.jpg"  # Change this to your desired output path
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    # Run inference
    inference_image(
        model_path=model_path,
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        conf_threshold=0.25  # Adjust this value to filter detections by confidence
    )

