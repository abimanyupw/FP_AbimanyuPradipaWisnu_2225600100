"""
Baseline training script - PENSHIP Recruitment Final Project

Flow:
  1. Download the dataset from the participant's Roboflow project (YOLOv8 format).
     If dataset/ already contains data.yaml, the download is skipped.
  2. Train YOLOv8.
  3. Evaluate the best model (best.pt) on the validation split.

Setup:
  1. Fill in ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION in the
     CONFIGURATION section. All three appear in the dataset URL:
         app.roboflow.com/<workspace>/<project>/<version>
     or in the snippet under Download Dataset -> YOLOv8 format -> Show download code.
  2. Set the Roboflow API key as an environment variable. Do NOT write the API
     key in this file, because this file is part of the submission.

Run (replace API_KEY_ROBOFLOW with your Roboflow API key):
  Windows (PowerShell):
      pip install -r requirements.txt
      $env:ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

  Linux / WSL / macOS:
      pip install -r requirements.txt
      export ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

For training on Google Colab, use train_colab.ipynb.

This code is intentionally minimal and MUST be extended by each participant.
See the README and the guidebook (Pengembangan Kode section) for ideas.
"""

import os
from pathlib import Path

from ultralytics import YOLO

# ==================== CONFIGURATION ====================
# Dataset (Roboflow)
ROBOFLOW_WORKSPACE = "workspace-name"  # from the dataset URL
ROBOFLOW_PROJECT = "project-name"      # from the dataset URL
ROBOFLOW_VERSION = 1                   # dataset version number on Roboflow
DATASET_DIR = "dataset"                # download destination folder

# Training
MODEL = "yolov8n.pt"  # options: yolov8n.pt or yolov8s.pt
EPOCHS = 50
IMGSZ = 640           # must stay 640, do not change
BATCH = 16            # lower to 8 or 4 on out-of-memory errors
SEED = 0
# =======================================================


def get_dataset() -> Path:
    """Return the path to data.yaml, downloading from Roboflow if it is missing."""
    data_yaml = Path(DATASET_DIR) / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset already exists, skipping download: {data_yaml}")
        return data_yaml

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. See the 'Run' section at the top of this file."
        )

    from roboflow import Roboflow  # imported here: not needed if the dataset already exists

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8", location=DATASET_DIR)

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found after download: {data_yaml}")
    return data_yaml


def main():
    data_path = get_dataset()

    # Pretrained model (weights are downloaded automatically on first use)
    model = YOLO(MODEL)

    # Training. Uses the GPU automatically if available, otherwise the CPU.
    model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
    )

    # Training output folder (runs/detect/train, train2, ...)
    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_path}")

    # Evaluate the best model on the validation split
    best_model = YOLO(str(best_path))
    metrics = best_model.val(data=str(data_path), imgsz=IMGSZ)
    print(f"mAP50-95: {metrics.box.map:.3f} | mAP50: {metrics.box.map50:.3f}")


# Required guard, especially on Windows (the dataloader uses multiprocessing)
if __name__ == "__main__":
    main()
