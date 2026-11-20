"""AquaVision - Utility Functions for River Waste Detection
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from ultralytics import YOLO

# Standard class mapping verified from data.yaml
CLASS_NAMES: Dict[int, str] = {
    0: "can",
    1: "foam",
    2: "plastic",
    3: "plastic bottle",
    4: "unknow",
}

# Distinct BGR color palette for visualization
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 165, 255),    # can: Orange
    1: (255, 255, 0),    # foam: Cyan
    2: (0, 0, 255),      # plastic: Red
    3: (0, 255, 0),      # plastic bottle: Green
    4: (180, 105, 255),  # unknow: Pink / Purple
}


def get_default_weights_path() -> Path:
    """Return default path to best.pt weights file."""
    # Look in weights/ relative to project root
    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "weights" / "best.pt"
    if default_path.exists():
        return default_path
    
    # Fallback to local directory
    local_weights = Path("weights") / "best.pt"
    if local_weights.exists():
        return local_weights.resolve()

    return default_path


def load_model(weights_path: Optional[Union[str, Path]] = None) -> YOLO:
    """Load YOLO11n AquaVision model with error handling.

    Args:
        weights_path: Path to .pt file. If None, uses default weights/best.pt.

    Returns:
        Loaded Ultralytics YOLO model instance.
    """
    if weights_path is None:
        weights_path = get_default_weights_path()
    else:
        weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at '{weights_path}'. "
            "Please ensure weights/best.pt exists or pass a valid --weights argument."
        )

    model = YOLO(str(weights_path))
    return model


def draw_custom_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    class_id: int,
    confidence: float,
    class_name: Optional[str] = None,
) -> np.ndarray:
    """Draw styled bounding box and label badge on image.

    Args:
        image: BGR numpy image array.
        box: (x1, y1, x2, y2) pixel coordinates.
        class_id: Class integer index.
        confidence: Prediction confidence float (0-1).
        class_name: Optional label override.

    Returns:
        Annotated image array.
    """
    x1, y1, x2, y2 = map(int, box)
    color = CLASS_COLORS.get(class_id, (0, 255, 255))
    name = class_name or CLASS_NAMES.get(class_id, f"Class {class_id}")
    label = f"{name} {confidence:.1%}"

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    # Calculate text size for badge background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw label badge
    badge_y1 = max(0, y1 - text_h - 8)
    badge_y2 = y1
    badge_x2 = min(image.shape[1], x1 + text_w + 8)

    cv2.rectangle(image, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 4, badge_y2 - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )

    return image
