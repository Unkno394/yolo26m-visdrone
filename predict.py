"""Run the trained VisDrone detector on one image."""

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to an image")
    parser.add_argument("--weights", type=Path, default=Path(__file__).parent / "weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error("веса не найдены: скачайте yolo26m_test640.zip из Releases и распакуйте в корень проекта")

    model = YOLO(str(args.weights))
    result = model.predict(
        source=str(args.image), imgsz=args.imgsz, conf=args.conf,
        project=str(args.output.parent), name=args.output.name,
        exist_ok=True, save=True,
    )[0]
    counts = Counter(model.names[int(cls)] for cls in result.boxes.cls.tolist())
    print(f"Detections: {dict(counts)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
