"""AquaVision - Image Waste Detection CLI

Detect floating waste in a single image or directory of images using YOLO11n.
"""

import argparse
from pathlib import Path
import sys
import cv2

# Add parent directory to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import CLASS_NAMES, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaVision YOLO11n - Detect river surface waste in images."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image file or directory of images.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained weights file (default: weights/best.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_outputs/images",
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detection results in an OpenCV window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', '0', '1', etc. (auto-detects if None).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"Error: Source '{args.source}' does not exist.")
        sys.exit(1)

    print("==================================================")
    print("   AquaVision YOLO11n - River Waste Detection     ")
    print("==================================================")
    print(f"Loading model weights...")
    model = load_model(args.weights)
    print(f"Model loaded successfully. Classes: {list(model.names.values())}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Perform inference
    kwargs = {
        "source": str(source_path),
        "conf": args.conf,
        "iou": args.iou,
        "save": True,
        "project": str(output_dir.parent),
        "name": output_dir.name,
        "exist_ok": True,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    print(f"Running inference on: {source_path}")
    results = model.predict(**kwargs)

    total_detections = 0
    class_counts = {name: 0 for name in model.names.values()}

    for i, r in enumerate(results):
        num_boxes = len(r.boxes)
        total_detections += num_boxes
        print(f"\n[{i+1}/{len(results)}] {Path(r.path).name}: {num_boxes} debris items detected")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            print(f"  - {cls_name:<16} (Confidence: {confidence:.1%}) at [{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]")

        if args.show:
            annotated_frame = r.plot()
            cv2.imshow("AquaVision Detection", annotated_frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    print("\n---------------- SUMMARY -----------------")
    print(f"Total images processed : {len(results)}")
    print(f"Total debris detected  : {total_detections}")
    print("Class breakdown        :")
    for cls_name, count in class_counts.items():
        if count > 0:
            print(f"  * {cls_name:<16}: {count}")
    print(f"Annotated outputs saved: {output_dir.resolve()}")
    print("==========================================")


if __name__ == "__main__":
    main()
