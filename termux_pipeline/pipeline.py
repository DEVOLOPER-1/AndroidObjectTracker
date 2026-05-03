from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_orchestrator import PipelineConfig, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLO first-frame detection and AbaVi tracking over a video.")
    parser.add_argument("video_path", type=Path, help="Path to the input video")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Path to the saved output video")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window")
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Process every Nth frame with the tracker (undersampling).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=200,
        help="Maximum number of trajectory points kept for drawing.",
    )
    parser.add_argument(
        "--trail-min-alpha",
        type=float,
        default=0.25,
        help="Minimum intensity factor for oldest trail segments (0.0-1.0).",
    )
    parser.add_argument(
        "--preferred-class",
        type=str,
        default="car",
        help="Preferred YOLO class to initialize tracking with.",
    )
    parser.add_argument(
        "--max-search-frames",
        type=int,
        default=30,
        help="Maximum initial frames scanned to find the first YOLO detection.",
    )
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=args.video_path,
        output_path=args.output,
        display=not args.no_display,
        process_every_n=args.process_every,
        preferred_class=args.preferred_class,
        max_search_frames=args.max_search_frames,
        trail_length=args.trail_length,
        trail_min_alpha=args.trail_min_alpha,
    )

    output_path = run_pipeline(config)
    print(f"Saved tracked video to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# python pipeline.py 20260503_125634.mp4  --no-display --process-every 3 --trail-length 300 --trail-min-alpha 0.2