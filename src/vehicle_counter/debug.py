"""
Visual debugger module.
Renders YOLO tracking bounding boxes and the spatial calibration line
over the raw video frames to validate spatial alignment in real-time.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from vehicle_counter.io.config import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Decoupled data directory. Respects container environments or local fallbacks.
DATA_DIR = Path(os.getenv("VEHICLE_DATA_DIR", Path.cwd() / "data/raw"))


def run(
    model_path: Path,
    context_name: Path,
) -> None:
    """
    Orchestrates the OpenCV rendering loop over the YOLO inference stream.
    Validates physical inputs before allocating the model to memory.
    """

    context_path = DATA_DIR / context_name
    video_path = context_path / context_name.with_suffix(".mp4")
    json_file = context_path / context_name.with_suffix(".json")

    # Strict Physical Validation (Fail Fast)
    if not video_path.is_file():
        msg = f"Critical Error: Video file not found at {video_path}"
        raise FileNotFoundError(msg)

    if not json_file.is_file():
        msg = (
            f"Critical Error: Calibration matrix not found at {json_file}. "
            f"Run 'uv run vehicle-calibrate --context {context_name}' first."
        )
        raise FileNotFoundError(msg)

    line_points = load_config(json_file)

    if line_points is None or len(line_points) != 2:
        msg = (
            "Critical Error: "
            f"Calibration file at {json_file} is corrupted or incomplete."
        )
        raise ValueError(msg)

    model_yolo = YOLO(model_path)

    results = model_yolo.track(
        source=str(video_path),
        stream=True,
        conf=0.5,
        imgsz=640,
    )

    for result in results:
        frame = result.orig_img

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 2)

        cv2.imshow("Visual Debugger", frame)
        key = cv2.waitKey(1) & 0xFF

        # Break loop on 'q' or 'ESC' key press
        if key in (ord("q"), 27):
            break

        # Break loop if the user forcibly closes the OS window
        if cv2.getWindowProperty("Visual Debugger", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


def main() -> int:
    """
    Parses command-line arguments and triggers the visual debugger.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Visual debugger for YOLO tracking and calibration line verification"
        )
    )

    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        help="Name of the context folder. Resolved against $VEHICLE_DATA_DIR.",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to the YOLO model weights file (e.g., yolov8n.pt).",
        required=True,
    )

    args = parser.parse_args()

    if not args.model.exists():
        logger.error("Model weights file not found.")
        return 1

    try:
        run(args.model, args.context)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Shutting down gracefully...")
        return 1
    except (FileNotFoundError, ValueError) as e:
        # Graceful handling for known domain errors to avoid
        # raw stack traces on the terminal
        logger.error("%s", e)  # noqa: TRY400
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
