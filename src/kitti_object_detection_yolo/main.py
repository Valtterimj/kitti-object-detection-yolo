from pathlib import Path
from collections import Counter
from ultralytics import YOLO

import sys

def main():

    print(sys.path)
    # print("Reading KITTI data")
    # PROJECT_ROOT = Path(__file__).resolve().parents[2]
    # DATA_YAML = PROJECT_ROOT / "kitti.yaml"

    # model = YOLO("yolo11s.pt")

    # model.train(
    #     data=str(DATA_YAML),
    #     epochs=50,
    #     imgsz=640,
    #     batch=8,
    #     device=0,
    #     project=str(PROJECT_ROOT / "runs"),
    #     name="yolo11s_kitti",
    #     pretrained=True,
    # )

if __name__ == "__main__":
    main()
