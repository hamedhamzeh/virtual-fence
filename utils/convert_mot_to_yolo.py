"""
Convert MOT17 / MOT20 annotations to YOLOv8 format with frame skipping.

You can:
- Choose which sequences go to train/val/test by editing the lists below.
- Control frame skipping via FRAME_STRIDE (default: keep every 10th frame).
- Create a new YOLOv8-style dataset folder only for MOT data.

Output structure (under OUTPUT_ROOT):
- train/images, train/labels
- val/images,   val/labels
- test/images,  test/labels

Labels are YOLO: <class_id> <x_center> <y_center> <width> <height> (all normalized).
"""

from __future__ import annotations

import csv
import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm  # pip install tqdm
import shutil


# ---------------------------------------------------------------------------
# Configuration – EDIT THESE
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "Datasets"

MOT17_ROOT = DATASETS_ROOT / "Mot17"
MOT20_ROOT = DATASETS_ROOT / "Mot20"

# New dataset root for YOLO-formatted MOT data
OUTPUT_ROOT = DATASETS_ROOT / "MOT_yolo"

# Keep every N-th frame (Option A from your description)
FRAME_STRIDE = 10

# Sequence splits – fill these lists with folder names (e.g., "MOT17-02").
# You can leave any list empty if you don't want to use that split.

# MOT17
MOT17_TRAIN_SEQS_TRAIN: List[str] = [
    "MOT17-04",
    "MOT20-03",
    "MOT20-05",
]

MOT17_TRAIN_SEQS_VAL: List[str] = [
]

MOT17_TRAIN_SEQS_TEST: List[str] = [
    # "MOT17-09",
]

# MOT20
MOT20_TRAIN_SEQS_TRAIN: List[str] = [
    "MOT20-03",
    "MOT20-05",
]

MOT20_TRAIN_SEQS_VAL: List[str] = [
     "MOT20-02",
]

MOT20_TRAIN_SEQS_TEST: List[str] = [
     "MOT20-01",
]

# Class filtering:
# Official MOT GT columns:
# frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
#
# Set allowed classes per dataset; if None, keep all rows with conf > 0.
# For "pedestrians only", typically class_id == 1. You can adjust if needed.

MOT17_ALLOWED_CLASSES: Optional[List[int]] = [1]  # set to None to keep all valid boxes
MOT20_ALLOWED_CLASSES: Optional[List[int]] = [1]

# Single YOLO class index used in output (0 = "People")
YOLO_CLASS_ID = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class SeqInfo:
    name: str
    img_dir: Path
    frame_rate: int
    seq_length: int
    img_width: int
    img_height: int
    img_ext: str


def load_seq_info(seq_dir: Path) -> SeqInfo:
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"Missing seqinfo.ini in {seq_dir}")

    config = ConfigParser()
    config.read(ini_path)

    seq_section = config["Sequence"]
    name = seq_section.get("name")
    im_dir = seq_section.get("imDir", "img1")
    frame_rate = seq_section.getint("frameRate")
    seq_length = seq_section.getint("seqLength")
    im_width = seq_section.getint("imWidth")
    im_height = seq_section.getint("imHeight")
    im_ext = seq_section.get("imExt", ".jpg")

    img_dir = seq_dir / im_dir

    return SeqInfo(
        name=name,
        img_dir=img_dir,
        frame_rate=frame_rate,
        seq_length=seq_length,
        img_width=im_width,
        img_height=im_height,
        img_ext=im_ext,
    )


def parse_gt(
    gt_path: Path,
    allowed_classes: Optional[List[int]],
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    Parse MOT GT file and return mapping frame -> list of (x, y, w, h) in pixels.
    Only rows with conf > 0 and (if specified) class in allowed_classes are kept.
    """
    frame_boxes: Dict[int, List[Tuple[float, float, float, float]]] = {}

    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    with gt_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue

            try:
                frame = int(row[0])
                # id_ = int(row[1])  # unused
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                conf = float(row[6])
                class_id: Optional[int] = None
                if len(row) >= 8:
                    try:
                        class_id = int(float(row[7]))
                    except ValueError:
                        class_id = None
            except ValueError:
                # Malformed line – skip
                continue

            # Keep only "true" annotations
            if conf <= 0:
                continue

            if allowed_classes is not None:
                if class_id is None or class_id not in allowed_classes:
                    continue

            frame_boxes.setdefault(frame, []).append((x, y, w, h))

    return frame_boxes


def yolo_from_xywh_pixels(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert top-left (x, y, w, h) in pixels to normalized YOLO (xc, yc, wn, hn)."""
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h

    # Clamp to [0, 1] just in case of rounding or slightly out-of-bounds boxes
    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    return clamp(xc), clamp(yc), clamp(wn), clamp(hn)


def ensure_split_dirs(split: str) -> Tuple[Path, Path]:
    """Create images/labels dirs for a split and return their paths."""
    images_dir = OUTPUT_ROOT / split / "images"
    labels_dir = OUTPUT_ROOT / split / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def convert_sequence(
    dataset_name: str,
    seq_name: str,
    seq_dir: Path,
    split: str,
    allowed_classes: Optional[List[int]],
) -> Tuple[int, int]:
    """
    Convert one MOT sequence into YOLO format.

    Returns (num_frames_processed, num_boxes_total).
    """
    seq_info = load_seq_info(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    frame_to_boxes = parse_gt(gt_path, allowed_classes=allowed_classes)

    images_dir_out, labels_dir_out = ensure_split_dirs(split)

    # Get list of frames by scanning image files; safer than relying only on seqLength.
    img_files = sorted(
        p for p in seq_info.img_dir.glob(f"*{seq_info.img_ext}") if p.is_file()
    )

    num_frames = 0
    num_boxes_total = 0

    for img_path in tqdm(
        img_files,
        desc=f"{dataset_name} {seq_name} {split}",
        unit="frame",
        leave=False,
    ):
        try:
            frame_idx = int(img_path.stem)
        except ValueError:
            # Unexpected filename – skip
            continue

        # Apply frame skipping rule (keep every N-th frame)
        if FRAME_STRIDE > 1 and (frame_idx % FRAME_STRIDE != 0):
            continue

        boxes = frame_to_boxes.get(frame_idx, [])

        out_stem = f"{seq_name}_{frame_idx:06d}"
        out_img_path = images_dir_out / f"{out_stem}{img_path.suffix}"
        out_lbl_path = labels_dir_out / f"{out_stem}.txt"

        # Copy image
        shutil.copy2(img_path, out_img_path)

        # Write label file (even if empty)
        with out_lbl_path.open("w") as lf:
            for (x, y, w, h) in boxes:
                xc, yc, wn, hn = yolo_from_xywh_pixels(
                    x, y, w, h, seq_info.img_width, seq_info.img_height
                )
                lf.write(
                    f"{YOLO_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
                )
                num_boxes_total += 1

        num_frames += 1

    return num_frames, num_boxes_total


def convert_dataset(
    dataset_name: str,
    root: Path,
    splits: Dict[str, List[str]],
    allowed_classes: Optional[List[int]],
) -> None:
    """Convert multiple sequences for a dataset (MOT17 or MOT20)."""
    for split, seq_names in splits.items():
        if not seq_names:
            continue

        print(f"\n=== {dataset_name} - {split} ===")

        total_frames = 0
        total_boxes = 0

        for seq_name in seq_names:
            seq_dir = root / "train" / seq_name
            if not seq_dir.is_dir():
                print(f"  [WARN] Sequence folder not found: {seq_dir}")
                continue

            print(f"  Processing sequence: {seq_name}")
            n_frames, n_boxes = convert_sequence(
                dataset_name=dataset_name,
                seq_name=seq_name,
                seq_dir=seq_dir,
                split=split,
                allowed_classes=allowed_classes,
            )
            print(f"    -> frames kept: {n_frames}, boxes: {n_boxes}")
            total_frames += n_frames
            total_boxes += n_boxes

        print(
            f"Summary {dataset_name} {split}: frames={total_frames}, boxes={total_boxes}"
        )


def main() -> None:
    print("Output YOLO MOT dataset root:", OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # MOT17
    mot17_splits = {
        "train": MOT17_TRAIN_SEQS_TRAIN,
        "val": MOT17_TRAIN_SEQS_VAL,
        "test": MOT17_TRAIN_SEQS_TEST,
    }
    convert_dataset(
        dataset_name="MOT17",
        root=MOT17_ROOT,
        splits=mot17_splits,
        allowed_classes=MOT17_ALLOWED_CLASSES,
    )

    # MOT20
    mot20_splits = {
        "train": MOT20_TRAIN_SEQS_TRAIN,
        "val": MOT20_TRAIN_SEQS_VAL,
        "test": MOT20_TRAIN_SEQS_TEST,
    }
    convert_dataset(
        dataset_name="MOT20",
        root=MOT20_ROOT,
        splits=mot20_splits,
        allowed_classes=MOT20_ALLOWED_CLASSES,
    )

    print("\nDone.")
    print(
        "You can now create a data.yaml pointing train/val/test to "
        f"{OUTPUT_ROOT / 'train' / 'images'}, etc."
    )


if __name__ == "__main__":
    main()


