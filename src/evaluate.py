"""AquaVision - Model Evaluation Script

Computes Precision, Recall, mAP@50, mAP@50-95, and F1-Score on the validation set.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaVision - Evaluate YOLO11n model on validation set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights (default: weights/best.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to dataset YAML file (default: data.yaml).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate ('val' or 'test').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use ('0' for GPU, 'cpu' for CPU).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.weights)

    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent / "data.yaml"

    print("==================================================")
    print("      AquaVision YOLO11n - Model Evaluation       ")
    print("==================================================")
    print(f"Dataset configuration : {data_path}")
    print(f"Evaluating split      : {args.split}")

    metrics = model.val(
        data=str(data_path),
        split=args.split,
        device=args.device,
        workers=0,
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n========== MODEL PERFORMANCE ==========")
    print(f"Precision       : {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall          : {recall:.4f} ({recall * 100:.2f}%)")
    print(f"mAP@50          : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP@50-95       : {map5095:.4f} ({map5095 * 100:.2f}%)")
    print(f"F1 Score        : {f1:.4f} ({f1 * 100:.2f}%)")
    print("========================================\n")


if __name__ == "__main__":
    main()
