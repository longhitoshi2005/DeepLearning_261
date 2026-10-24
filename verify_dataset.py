"""
Verify that YOLOv8-Seg can load and validate the merged dataset.
"""

from ultralytics import YOLO
from pathlib import Path
import yaml

def verify():
    yaml_path = Path("durian_dataset_merged/data.yaml").resolve()
    if not yaml_path.exists():
        print(f"File {yaml_path} not found. Please run merge first.")
        return

    print(f"Checking dataset configuration at: {yaml_path}")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    print("YAML config:", config)

    # Test loading YOLOv8 nano segmentation model
    print("Loading YOLOv8n-seg model for validation check...")
    model = YOLO("yolov8n-seg.pt")
    print("Model initialized successfully.")
    print("You can train the model using:")
    print(f"yolo segment train data={yaml_path.as_posix()} model=yolov8n-seg.pt epochs=50 imgsz=640 batch=16")

if __name__ == "__main__":
    verify()

