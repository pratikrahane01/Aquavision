"""AquaVision - YOLO11n Training Pipeline

Reproduce training on the river surface waste dataset.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaVision - Train YOLO11n on River Waste Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to data.yaml dataset descriptor (default: data.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model checkpoint or architecture (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image resolution (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference/training device: '0' for GPU, 'cpu' for CPU.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent.parent / "data.yaml"

    print("==================================================")
    print("      AquaVision YOLO11n - Model Training         ")
    print("==================================================")
    print(f"Dataset config : {data_path}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch}")
    print(f"Resolution     : {args.imgsz}")

    if args.resume:
        print(f"Resuming from  : {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
    else:
        print(f"Base model     : {args.model}")
        model = YOLO(args.model)
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            optimizer="MuSGD",
            device=args.device,
            amp=True,
            close_mosaic=10,
            project="runs/detect",
            name="train",
        )

    print("Training finished! Final metrics on validation set:")
    model.val()


if __name__ == "__main__":
    main()
