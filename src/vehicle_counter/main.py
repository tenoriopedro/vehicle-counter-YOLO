import argparse
import logging
import os
import sys
import time
from pathlib import Path

from vehicle_counter.io.config import load_config
from vehicle_counter.io.sink import TelemetrySink
from vehicle_counter.io.video import get_video_fps
from vehicle_counter.services.traffic_analyzer import TrafficCounter
from vehicle_counter.vision.adapters import build_yolo_stream

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
DATA_DIR = Path(os.getenv("VEHICLE_DATA_DIR", Path.cwd() / "data/raw"))


def run(
    model_path: Path,
    context_name: Path,
    classes_to_count: list[int],
) -> None:
    """
    Orchestrates the real-time inference and telemetry streaming pipeline.
    Expects a strict workspace structure: data/raw/<context>/<context>.[mp4|json]
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

    # Dependency Loading
    fps = get_video_fps(video_path)
    line_points = load_config(json_file)

    if line_points is None or len(line_points) != 2:
        msg = (
            "Critical Error: "
            f"Calibration file at {json_file} is corrupted or incomplete."
        )
        raise ValueError(msg)

    start_time = time.time()

    # Establish the network connection to the Kafka broker
    sink = TelemetrySink()

    try:
        frames_of_objects = build_yolo_stream(model_path, video_path, classes_to_count)
        traffic = TrafficCounter(frames_of_objects, fps, sink, line_points)

        # This loop blocks the main thread. It consumes the generator infinitely
        # until the video ends or a fatal exception is raised.
        traffic.start_tracking()

    finally:
        # Absolute guarantee of network drain.
        # Flushes the Kafka buffer even if the YOLO stream
        # crashes or the user hits Ctrl+C.
        sink.close()

    duration = time.time() - start_time
    formatted_time = time.strftime(
        "%H hours %M minutes %S seconds", time.gmtime(duration)
    )

    logger.info("STREAMING COMPLETED - Execution Time: %s", formatted_time)


def main() -> int:
    """
    Parses command-line arguments and triggers the execution pipeline.
    """

    parser = argparse.ArgumentParser(
        description="""
        Pipeline for bidirectional
        vehicle extraction and counting using YOLO models, streaming directly to Kafka.
        """
    )

    parser.add_argument(
        "-c",
        "--context",
        type=Path,
        help="""
        Name of the context folder inside data/raw/.
        The system expects data/raw/<context>/<context>.mp4 and <context>.json to exist.
        """,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        help="Path to the YOLO model weights file (e.g., yolov8n.pt).",
        required=True,
    )
    parser.add_argument(
        "--cls",
        nargs="+",
        type=int,
        help="""
        List of COCO dataset class IDs to track.
        Default is 2 3 7 (2=car, 3=motorcycle, 7=truck).
        """,
        default=[2, 3, 7],
    )

    args = parser.parse_args()

    if not args.model.exists():
        logger.error("Model weights file not found.")
        return 1

    try:
        run(args.model, args.context, args.cls)

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
