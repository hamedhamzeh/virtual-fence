import time
import torch
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_YAML = "datasets/FinalDataSet/DatasetsPaths.yaml"          # test set yaml
IMGSZ = 640
WARMUP_ITERS = 30
TIMING_ITERS = 200

MODELS = {
    "yolov8n": "trained_models/best_N.pt",
    "yolov8s": "trained_models/best_S.pt",
    "yolov8m": "trained_models/best_M.pt",
}
# --------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

results = []

if __name__ == "__main__":
    def measure_fps(model):
        imgs = torch.randn(1, 3, IMGSZ, IMGSZ).to(device)

        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = model(imgs)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(TIMING_ITERS):
            _ = model(imgs)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        fps = TIMING_ITERS / elapsed
        latency = 1000 / fps
        vram = torch.cuda.max_memory_allocated() / 1024**2

        return fps, latency, vram

    # ---------------- RUN BENCHMARK ----------------
    for name, weight_path in MODELS.items():
        print(f"\nBenchmarking {name}")

        model = YOLO(weight_path)

        # Accuracy metrics
        metrics = model.val(
            data=DATA_YAML,
            imgsz=IMGSZ,
            split="test",
            device=device,
            workers=0,
            verbose=False
        )

        fps, latency, vram = measure_fps(model.model)

        results.append({
            "Model": name,
            "mAP@0.5": metrics.box.map50,
            "mAP@0.5:0.95": metrics.box.map,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
            "FPS": fps,
            "Latency (ms)": latency,
            "VRAM (MB)": vram,
            "Model Size (MB)": Path(weight_path).stat().st_size / 1024**2
        })

    df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(df)

    df.to_csv("benchmark_results.csv", index=False)

