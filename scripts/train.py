import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="v1", help="Name of the dataset and output")
    parser.add_argument("--size", type=str, default="n", help="YOLO model size: n/s/m/l/x")
    parser.add_argument("--epochs", type=int, default=50, help="Max number of epochs to train")
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs without improvement")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--device", type=str, default=None, help="Training device: cpu/0/mps")
    args = parser.parse_args()
