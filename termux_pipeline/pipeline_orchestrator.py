"""
pipeline_orchestrator.py — Full bootstrap tracking pipeline.

Execution flow
--------------
1.  Scan up to `max_search_frames` frames with YOLO to find the RC car.
2.  On the frame where the car is first detected ("Frame 0"):
      a.  Initialise AbaViTracker with the car bbox.
      b.  Initialise PinTracker with the same frame → detects all pins by colour.
3.  For every subsequent frame ("Frame N+"):
      a.  Every `process_every_n` frames: run AbaViTracker.track() for the car.
      b.  Every frame (cheap): run PinTracker.update() for fall detection.
      c.  Compose and write the annotated frame.

Pin fall detection is always run on every frame (not gated by process_every_n)
because it is a simple pixel-count operation (~0.2 ms/frame) and skipping frames
would delay fall-event detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm.auto import tqdm

from abavit_tracker import AbaViTracker
from annotation_renderer import annotate_tracking_frame
from pin_tracker import PinTracker
from yolo_tracker import YOLOTracker


@dataclass(slots=True)
class PipelineConfig:
    video_path: Path
    output_path: Path | None = None
    display: bool = True
    process_every_n: int = 1
    preferred_class: str = "car"
    max_search_frames: int = 30
    trail_length: int = 200
    trail_min_alpha: float = 0.25
    show_pin_roi: bool = False   # draw the fall-detection ROI boxes (debug)
    show_score: bool = True      # show score overlay with elapsed time + pin count


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (unchanged from original, kept for completeness)
# ─────────────────────────────────────────────────────────────────────────────

def _pick_initial_detection(detections, preferred_class: str):
    if not detections:
        return None
    preferred = [d for d in detections if getattr(d, "object_type", None) == preferred_class]
    pool = preferred if preferred else detections
    return max(pool, key=lambda d: float(getattr(d, "confidence", 0.0)))


def _find_initial_detection(cap, yolo: YOLOTracker, preferred_class: str, max_search_frames: int):
    for frame_index in range(max_search_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        detections = yolo.infer(frame_bgr, is_bgr=True)
        detection  = _pick_initial_detection(detections, preferred_class=preferred_class)
        if detection is not None:
            return frame_index, frame_bgr, detection
    return None, None, None


def _center_from_bbox(bbox: list[int] | tuple[int, int, int, int]) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    return [x + w // 2, y + h // 2]


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> Path:
    process_every_n   = max(1, int(config.process_every_n))
    max_search_frames = max(1, int(config.max_search_frames))
    trail_length      = max(2, int(config.trail_length))

    # ── Build trackers ───────────────────────────────────────────────────────
    yolo = YOLOTracker()

    params = type("P", (), {})()
    params.tracker_name = "abavit"
    params.param_name   = "default"
    abavit = AbaViTracker(params, dataset_name="custom")

    pin_tracker = PinTracker()   # ← new

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {config.video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    # ── Frame 0: find car with YOLO ──────────────────────────────────────────
    first_frame_index, first_frame_bgr, detection = _find_initial_detection(
        cap, yolo,
        preferred_class=config.preferred_class,
        max_search_frames=max_search_frames,
    )
    if first_frame_bgr is None or detection is None:
        cap.release()
        raise SystemExit("YOLO found no car in the first {max_search_frames} frames.")

    # ── Initialise car tracker ───────────────────────────────────────────────
    x1, y1, x2, y2 = detection.box
    init_bbox     = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    last_valid_bbox = init_bbox
    label           = getattr(detection, "object_type", "car")
    trajectory_points = [_center_from_bbox(init_bbox)]

    first_frame_rgb = cv.cvtColor(np.asarray(first_frame_bgr), cv.COLOR_BGR2RGB)
    abavit.initialize(first_frame_rgb, {"init_bbox": init_bbox})

    # ── Initialise pin tracker (same frame 0) ─────────────────────────────────
    #
    #   We pass the car's bbox as an exclude region so that, even if the user
    #   swaps to a coloured car, the car body pixels are masked out during pin
    #   initialisation.  For the user's white car this has no effect.
    pin_states = pin_tracker.initialize(
        first_frame_rgb,
        exclude_regions=[init_bbox],   # mask the car footprint on frame 0
    )
    print(f"[Pipeline] Frame 0: {len(pin_states)} pins detected, car at {init_bbox}")

    # ── Output video writer ───────────────────────────────────────────────────
    output_path = config.output_path
    if output_path is None:
        output_path = config.video_path.with_name(
            f"{config.video_path.stem}_tracked.mp4"
        )
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)  or first_frame_bgr.shape[1])
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or first_frame_bgr.shape[0])
    writer = cv.VideoWriter(
        str(output_path),
        getattr(cv, "VideoWriter_fourcc")(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    window_name = "AbaViTrack + PinTracker"
    if config.display:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    total_frames   = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    progress_total = total_frames if total_frames > 0 else None
    pbar = tqdm(
        total=progress_total,
        desc=f"Tracking {config.video_path.name}",
        unit="frame",
    )

    pipeline_start_time = time.time()

    try:
        if first_frame_index is not None:
            pbar.update(first_frame_index + 1)

        # ── Annotate and write frame 0 ─────────────────────────────────────
        elapsed = time.time() - pipeline_start_time
        first_annotated = annotate_tracking_frame(
            np.asarray(first_frame_bgr),
            last_valid_bbox,
            label,
            trajectory_points,
            trail_length      = trail_length,
            trail_min_alpha   = config.trail_min_alpha,
            pin_states        = pin_states,
            elapsed_seconds   = elapsed if config.show_score else None,
        )
        writer.write(first_annotated)

        if config.display:
            cv.imshow(window_name, first_annotated)
            if cv.waitKey(1) & 0xFF == ord("q"):
                return output_path

        # ── Main loop: frames 1 … end ─────────────────────────────────────
        frame_index = (first_frame_index or 0) + 1
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

            # ── Car tracking (skipped on non-process frames) ───────────────
            should_process = (
                (frame_index - (first_frame_index or 0)) % process_every_n
            ) == 0

            if should_process:
                out_car = abavit.track(frame_rgb)
                tracked_bbox = (
                    out_car.get("target_bbox")
                    if isinstance(out_car, dict) else None
                )
                if (
                    isinstance(tracked_bbox, (list, tuple))
                    and len(tracked_bbox) == 4
                ):
                    last_valid_bbox = [int(v) for v in tracked_bbox]
                    trajectory_points.append(_center_from_bbox(last_valid_bbox))

            # ── Pin fall detection (every frame — very cheap) ──────────────
            #
            #   PinTracker.update() only iterates over still-standing pins and
            #   counts coloured pixels in small fixed ROIs.  It does not run
            #   any neural network inference and adds < 1 ms per frame.
            pin_states = pin_tracker.update(frame_rgb)

            # ── Compose annotated frame ────────────────────────────────────
            elapsed = time.time() - pipeline_start_time
            annotated = annotate_tracking_frame(
                frame_bgr,
                last_valid_bbox,
                label,
                trajectory_points,
                trail_length    = trail_length,
                trail_min_alpha = config.trail_min_alpha,
                pin_states      = pin_states,
                elapsed_seconds = elapsed if config.show_score else None,
            )

            writer.write(annotated)
            pbar.update(1)

            if config.display:
                cv.imshow(window_name, annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        # ── Final stats ────────────────────────────────────────────────────
        total_elapsed = time.time() - pipeline_start_time
        fallen = pin_tracker.fallen_count()
        fallen_pins = sorted(
            [p for p in pin_tracker.get_pins() if p.is_fallen],
            key=lambda p: p.fall_order,
        )
        print(
            f"\n[Pipeline] Done in {total_elapsed:.1f}s — "
            f"{fallen}/{len(pin_tracker.get_pins())} pins knocked down"
        )
        if fallen_pins:
            order_str = " → ".join(
                f"#{p.fall_order}({p.color_name})" for p in fallen_pins
            )
            print(f"[Pipeline] Fall order: {order_str}")

        return output_path

    finally:
        cap.release()
        writer.release()
        pbar.close()
        if config.display:
            cv.destroyAllWindows()
