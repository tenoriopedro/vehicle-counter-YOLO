import argparse
import logging
import os
import sys
from pathlib import Path

from vehicle_counter.io.config import save_config
from vehicle_counter.io.video import load_frame
from vehicle_counter.presentation.draw_lines import LineCalibrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("VEHICLE_DATA_DIR", Path.cwd() / "data/raw"))


def main() -> int:
    """
    CLI entry point for the video calibration tool.
    Extracts a frame, launches the UI, and saves the configuration.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Interactive tool to calibrate the counting line for a specific video."
        )
    )
    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        help="Name of the context folder. Resolved against $VEHICLE_DATA_DIR",
        required=True,
    )

    args = parser.parse_args()

    context_path = DATA_DIR / args.context
    video_path = context_path / args.context.with_suffix(".mp4")
    json_file = context_path / args.context.with_suffix(".json")

    # Strict Physical Validation (Fail Fast)
    if not video_path.is_file():
        logger.error("Critical Error: Video file not found at %s", video_path)
        return 1

    frame = load_frame(video_path)

    calibrator = LineCalibrator(window_name=f"Calibration: {args.context.name}")
    line_points = calibrator.run(frame)  # type: ignore[reportArgumentType]

    if len(line_points) != 2:
        logger.error(
            "Calibration aborted or incomplete. Exactly two points are required."
        )
        return 1

    save_config(json_file, line_points)
    logger.info("Calibration successful! Configuration saved to '%s'", json_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
