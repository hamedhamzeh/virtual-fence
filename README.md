# Virtual Fence & Person Tracking

This project implements a computer vision pipeline for detecting and tracking people in video streams using YOLOv8 and a custom SimpleSORT tracker. It is designed for applications like virtual fencing and person counting.

<div align="center">
  <a href="URL_HERE">
    <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/c96f8a6d4e9ef64d52f287d6369a60052aa866a0/results/Demo.gif"
         alt="Demo"
         width="100%"/>
  </a>
</div>


## Features

- **Object Detection**: Uses YOLOv8 for high-performance person detection.
- **Object Tracking**: Implements SimpleSORT (Simple Online and Realtime Tracking) to maintain object identities across frames.
- **Zone and Counting**: Defines specific zones to count people entering or staying within them, useful for occupancy monitoring.
- **Visualizations**: Draws bounding boxes, track IDs, and zone counters on the output video.

## Project Structure

- `inference.py`: Main script to run the detection and tracking pipeline.
- `tracker.py`: Implementation of the SimpleSORT tracker logic.
- `visualizer.py`: Utility for drawing tracks and detection boxes on frames.
- `counter_zone.py`: Implements logic for zone-based counting.
- `benchmark.py`: Script to benchmark YOLOv8 models.
- `utils/`: Helper scripts.
- `results/`: Directory for storing output videos and results.
- `trained_models/`: Directory for storing YOLOv8 model weights (e.g., `best-40.pt`).

## Training Pipeline and Aggregated dataset

- **Google drive link**: [Dataset Folder](https://drive.google.com/drive/folders/1SRhG4wKo4ld1BauXciEkSQVdjiEp-Ps3?usp=sharing)
- **Dataset Sources**:
    1. [MOT20 & MOT17Det](https://motchallenge.net)
    2. [top-view-multi-person-tracking](https://github.com/ucuapps/top-view-multi-person-tracking)
    3. [Roboflow Counting Dataset](https://app.roboflow.com/vcloudai/counting-2jyrc/4)
    4. [Roboflow Crowd Dataset](https://universe.roboflow.com/crowd-zuf0q/crowd-5ropi/dataset/1)
    5. [Roboflow Crowd Evaluation](https://universe.roboflow.com/crowd-counting-ssra0/crowd-counting-gdq7w/dataset/2)

| Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| :---: | :---: | :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/89378a46cb8cbea687997d4b704a32cf29c90606/dataset_samples/CCv4_MoT.png" width="200"> | <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/89378a46cb8cbea687997d4b704a32cf29c90606/dataset_samples/CCv3_topView.png" width="200"> | <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/89378a46cb8cbea687997d4b704a32cf29c90606/dataset_samples/CCv2i.png" width="200"> | <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/89378a46cb8cbea687997d4b704a32cf29c90606/dataset_samples/CCv1i.png" width="200"> | <img src="https://raw.githubusercontent.com/hamedhamzeh/virtual-fence/89378a46cb8cbea687997d4b704a32cf29c90606/dataset_samples/CCv0i.png" width="200"> |


## Benchmarks

Performance metrics for different YOLOv8 models on the test set.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS | Latency (ms) | VRAM (MB) | Model Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n** | 0.823 | 0.455 | 0.881 | 0.724 | 168.35 | 5.94 | 411.84 | 11.78 |
| **YOLOv8s** | 0.871 | 0.499 | 0.910 | 0.811 | 66.14 | 15.12 | 745.48 | 21.48 |
| **YOLOv8m** | 0.887 | 0.522 | 0.909 | 0.828 | 28.19 | 35.47 | 1228.87 | 49.63 |

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
