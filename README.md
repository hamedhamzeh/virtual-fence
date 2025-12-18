# Virtual Fence & Person Tracking

This project implements a computer vision pipeline for detecting and tracking people in video streams using YOLOv8 and a custom SimpleSORT tracker. It is designed for applications like virtual fencing and person counting.

## Features

- **Object Detection**: Uses YOLOv8 for high-performance person detection.
- **Object Tracking**: Implements SimpleSORT (Simple Online and Realtime Tracking) to maintain object identities across frames.
- **Visualizations**: Draws bounding boxes and track IDs on the output video.
- **Filtering**: (TODO) Capabilities to filter detections based on box area, aspect ratio, etc.

## Project Structure

- `inference.py`: Main script to run the detection and tracking pipeline.
- `tracker.py`: Implementation of the SimpleSORT tracker logic.
- `visualizer.py`: Utility for drawing tracks and detection boxes on frames.
- `filters.py`: (TODO) Contains logical filters for refining detection results.
- `results/`: Directory for storing output videos and results.
- `utils/`: Helper functions.
- `trained_models/`: Directory for storing YOLOv8 model weights (e.g., `best-40.pt`).

## Training Pipeline and Aggregated dataset:

- Google drive link: https://drive.google.com/drive/folders/1SRhG4wKo4ld1BauXciEkSQVdjiEp-Ps3?usp=sharing
- Dataset is the combination of these sources:
    . https://motchallenge.net/data/MOT20/
    . https://motchallenge.net/data/MOT17Det/
    . https://app.roboflow.com/vcloudai/counting-2jyrc/4
    . https://universe.roboflow.com/crowd-zuf0q/crowd-5ropi/dataset/1
    . https://universe.roboflow.com/crowd-counting-ssra0/crowd-counting-gdq7w/dataset/2

## Prerequisites

- Python 3.8+
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV
- PyTorch
- NumPy

## Installation

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

*(Note: Ensure your `requirements.txt` includes `ultralytics`, `opencv-python`, `torch`, `torchvision`, `numpy`)*

## Usage

1. Place your input video file in the project root (e.g., `input.mp4`).
2. Ensure you have a trained model in `trained_models/` or update the `model_path` in `inference.py`.
3. Run the inference script:

```bash
python inference.py
```

The script will process the video and generate an output file (default: `output_tracked_simple_sort.mp4`) with overlaid tracking visualizations.

## Configuration

You can adjust parameters in `inference.py` such as:
- `conf_threshold`: Confidence threshold for detections.
- `iou_threshold`: IOU threshold for NMS.
- `tracker`: Settings for the SimpleSORT tracker (max age, min/max box area).

## License

[Add your license here]
