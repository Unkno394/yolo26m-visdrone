"""Reproduce the later 640px, 10-epoch YOLO26m VisDrone run."""

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="VisDrone.yaml")
    parser.add_argument("--device", default="0", help="GPU index, or cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/train")
    args = parser.parse_args()

    model = YOLO("yolo26m.pt")
    model.train(
        data=args.data, epochs=10, imgsz=640, batch=args.batch,
        device=args.device, workers=4, amp=True, optimizer="auto",
        seed=42, plots=True, save=True, save_period=5,
        project=args.project, name="yolo26m_test640",
    )


if __name__ == "__main__":
    main()
