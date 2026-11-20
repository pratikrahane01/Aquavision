"""AquaVision - Video & Webcam Waste Detection CLI

Detect floating river debris in recorded video files or live webcam streams.
"""

import argparse
from pathlib import Path
import sys
import time
import cv2

# Add parent directory to path to allow direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="AquaVision YOLO11n - Detect river waste in video or live camera."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video file path or webcam index (default: '0' for default webcam).",
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
        default="sample_outputs/videos/output_video.mp4",
        help="Output video file path (for video files).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display live preview window (default: True).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', '0', etc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print("   AquaVision YOLO11n - Video Waste Detection     ")
    print("==================================================")
    print("Loading model weights...")
    model = load_model(args.weights)
    print(f"Model loaded. Classes: {list(model.names.values())}")

    # Determine source type
    is_cam = args.source.isdigit() or args.source == "0"
    source = int(args.source) if is_cam else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    video_writer = None
    if not is_cam and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"Saving output video to: {out_path}")

    print("Starting detection loop (press 'q' to quit)...")
    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_infer = time.time()

            kwargs = {
                "source": frame,
                "conf": args.conf,
                "iou": args.iou,
                "verbose": False,
            }
            if args.device is not None:
                kwargs["device"] = args.device

            results = model.predict(**kwargs)
            infer_ms = (time.time() - start_infer) * 1000

            annotated_frame = results[0].plot()

            # Add FPS and count overlay
            current_fps = frame_count / max(time.time() - t0, 1e-3)
            num_debris = len(results[0].boxes)
            overlay_text = f"AquaVision | FPS: {current_fps:.1f} ({infer_ms:.1f}ms) | Debris: {num_debris}"
            cv2.putText(
                annotated_frame,
                overlay_text,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if args.show:
                cv2.imshow("AquaVision Stream Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Detection terminated by user.")
                    break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    avg_fps = frame_count / max(elapsed, 1e-3)
    print("\n---------------- SUMMARY -----------------")
    print(f"Total frames processed : {frame_count}")
    print(f"Average processing FPS : {avg_fps:.2f}")
    if video_writer is not None:
        print(f"Output saved to        : {args.output}")
    print("==========================================")


if __name__ == "__main__":
    main()
