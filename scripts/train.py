import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings

settings.update({'mlflow': False})  # disable local file-store crash

def resolve_device(override: str | None = None) -> int | str:
    if override:
        print(f"Training device: {override} (override)")
        return override
    elif torch.cuda.is_available():
        print(f"Training device: {torch.cuda.get_device_name(0)}")
        return 0
    elif torch.mps.is_available():
        print("Training device: Apple Silicon MPS")
        return 'mps'
    print("Training device: CPU")
    return 'cpu'

def train_model():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="v1", help="Name of the dataset and output")
    parser.add_argument("--size", type=str, default="n", help="YOLO model size: n/s/m/l/x")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs to train")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs without improvement")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--device", type=str, default=None, help="Training device: cpu/0/mps")
    args = parser.parse_args()
