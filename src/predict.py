"""AquaVision - Unified Prediction Entry Point

Run YOLO11n river waste inference on images, video files, directories, or webcam.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaVision - YOLO11n River Surface Waste Predictor"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: image path, video path, folder path, or '0' for live webcam.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained YOLO11n weights file (default: weights/best.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save prediction results (default: True).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction results interactively.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu', '0', etc.).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sample_outputs",
        help="Directory to save prediction results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predictions",
        help="Subdirectory name for run outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("      AquaVision YOLO11n - Waste Detection       ")
    print("==================================================")
    model = load_model(args.weights)
    print(f"Model loaded: {args.weights or 'weights/best.pt'}")
    print(f"Recognized classes: {model.names}")

    kwargs = {
        "source": args.source if not args.source.isdigit() else int(args.source),
        "conf": args.conf,
        "iou": args.iou,
        "save": args.save,
        "show": args.show,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    print(f"\nRunning prediction on '{args.source}' (conf={args.conf})...")
    results = model.predict(**kwargs)

    print("\nInference complete!")
    if args.save:
        save_path = Path(args.project) / args.name
        print(f"Results saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
