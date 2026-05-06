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


import annotation_renderer
from pipeline import PinFallTracker, center_from_bbox, pick_primary_detection
from yolo_tracker import Detection, YOLOTracker, TrackedDetection


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
    nms_score_threshold: float = 0.25   # YOLO confidence filter (before NMS) — try 0.35-0.50 for stricter
    nms_iou_threshold: float = 0.45     # Overlap tolerance in NMS — lower (0.25-0.35) = more aggressive


def _xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[int]:
    x1, y1, x2, y2 = box
    return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]


def _select_pin_detections(
    detections: list[TrackedDetection],
) -> list[TrackedDetection]:
    """
    Filter to bowling-pins class and sort by confidence.

    NMS is already applied upstream in yolo_tracker.infer(), so NMS here is redundant.
    """
    pins = [d for d in detections if getattr(d, "object_type", None) == "bowling-pins"]
    return sorted(pins, key=lambda d: float(d.confidence), reverse=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(config: PipelineConfig) -> Path:
    process_every_n   = max(1, int(config.process_every_n))
    max_search_frames = max(1, int(config.max_search_frames))
    trail_length      = max(2, int(config.trail_length))

    # ── Build trackers ───────────────────────────────────────────────────────
    yolo = YOLOTracker()

    # ── Apply NMS thresholds from config ──────────────────────────────────
    yolo.nms_score_threshold = config.nms_score_threshold
    yolo.nms_iou_threshold = config.nms_iou_threshold
    print(f"[Pipeline] NMS thresholds: score={config.nms_score_threshold}, iou={config.nms_iou_threshold}")

    pin_tracker = PinFallTracker()

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {config.video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    # ── Frame 0: find car with YOLO ──────────────────────────────────────────
    first_frame_index: int | None = None
    first_frame_bgr: np.ndarray | None = None
    first_tracked: list[TrackedDetection] = []
    car_detection: Detection | TrackedDetection | None = None

    for frame_index in range(max_search_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        tracked_detections = yolo.track(frame_bgr, is_bgr=True)
        selected_car = pick_primary_detection(tracked_detections, config.preferred_class)
        if selected_car is not None:
            first_frame_index = frame_index
            first_frame_bgr = frame_bgr
            first_tracked = tracked_detections
            car_detection = selected_car
            break

    if first_frame_bgr is None or car_detection is None:
        cap.release()
        raise SystemExit("YOLO found no car in the first {max_search_frames} frames.")

    # ── STRICT CLASS VALIDATION: car must be class_id == 3 ─────────────────
    if int(getattr(car_detection, "class_id", -1)) != 3:
        cap.release()
        raise SystemExit(
            f"Frame {first_frame_index}: Invalid car detection. "
            f"Expected class_id=3, got class_id={car_detection.class_id} ({car_detection.object_type})"
        )

    # ── Initialise car tracker ───────────────────────────────────────────────
    init_bbox     = _xyxy_to_xywh(car_detection.box)
    last_valid_bbox = init_bbox
    label           = "car"
    trajectory_points = [center_from_bbox(init_bbox)]

    # ── Initialise pin tracker (same frame 0) ─────────────────────────────────
    pin_states = pin_tracker.initialize(
        _select_pin_detections(first_tracked),
        car_bbox=init_bbox,
        frame_index=first_frame_index or 0,
    )
    print(f"[Pipeline] Frame {first_frame_index}: {pin_tracker.total_pins()} pins detected, car at {init_bbox}")

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
    display_enabled = bool(config.display)
    if display_enabled:
        try:
            cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        except cv.error as exc:
            display_enabled = False
            print(f"[Pipeline] Display disabled: OpenCV HighGUI is unavailable ({exc})")

    total_frames   = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    progress_total = total_frames if total_frames > 0 else None
    frames_70_percent = max(1, int(total_frames * 0.7)) if total_frames > 0 else 1
    detected_pin_track_ids: set[int] = set()  # Track unique pin track_ids across 70% of video

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
        first_annotated = annotation_renderer.annotate_tracking_frame(
            np.asarray(first_frame_bgr),
            last_valid_bbox,
            label,
            trajectory_points,
            trail_length      = trail_length,
            trail_min_alpha   = config.trail_min_alpha,
            pin_states        = pin_states,
            elapsed_seconds   = elapsed if config.show_score else None,
            show_roi          = config.show_pin_roi,
        )
        writer.write(first_annotated)

        if display_enabled:
            cv.imshow(window_name, first_annotated)
            if cv.waitKey(1) & 0xFF == ord("q"):
                return output_path

        # ── Main loop: frames 1 … end ─────────────────────────────────────
        frame_index = (first_frame_index or 0) + 1
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            tracked_detections = yolo.track(frame_bgr, is_bgr=True)

            # ── Car tracking (trajectory updates can be undersampled) ───────
            # CRITICAL: Only accept class_id == 3 (car) to prevent pins from hijacking the car track
            car_candidates = [
                d for d in tracked_detections
                if int(getattr(d, "class_id", -1)) == 3 and getattr(d, "object_type", None) == config.preferred_class
            ]
            selected_car = pick_primary_detection(car_candidates, config.preferred_class)
            should_process = ((frame_index - (first_frame_index or 0)) % process_every_n) == 0
            if should_process and selected_car is not None:
                last_valid_bbox = _xyxy_to_xywh(selected_car.box)
                trajectory_points.append(center_from_bbox(last_valid_bbox))

            # ── Pin fall detection (every frame) ───────────────────────────
            car_bbox_for_fall = last_valid_bbox
            if selected_car is not None:
                car_bbox_for_fall = _xyxy_to_xywh(selected_car.box)
            pin_states = pin_tracker.update(
                _select_pin_detections(tracked_detections),
                car_bbox=car_bbox_for_fall,
                frame_index=frame_index,
            )

            # ── Track unique pins detected in first 70% of video ──────────────
            if frame_index < frames_70_percent:
                for pin in pin_states:
                    detected_pin_track_ids.add(pin.track_id)

            # ── Compose annotated frame ────────────────────────────────────
            elapsed = time.time() - pipeline_start_time
            annotated = annotation_renderer.annotate_tracking_frame(
                frame_bgr,
                last_valid_bbox,
                label,
                trajectory_points,
                trail_length    = trail_length,
                trail_min_alpha = config.trail_min_alpha,
                pin_states      = pin_states,
                elapsed_seconds = elapsed if config.show_score else None,
                show_roi        = config.show_pin_roi,
            )

            writer.write(annotated)
            pbar.update(1)

            if display_enabled:
                cv.imshow(window_name, annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        # ── Final stats ────────────────────────────────────────────────────
        total_elapsed = time.time() - pipeline_start_time
        fallen = pin_tracker.fallen_count()
        max_pins_detected = len(detected_pin_track_ids)
        fallen_pins = sorted([p for p in pin_tracker.get_pins() if p.is_fallen], key=lambda p: p.fall_order)

        # ── SANITY CHECK: Fallen count can't exceed detected pins ──────────
        if fallen > max_pins_detected:
            print(
                f"\n[WARNING] Sanity check failed: {fallen} pins marked as fallen, "
                f"but only {max_pins_detected} unique pins detected in first 70% of video."
            )
            fallen = max_pins_detected  # Cap to detected count

        print(
            f"\n[Pipeline] Done in {total_elapsed:.1f}s — "
            f"{fallen}/{max_pins_detected} pins knocked down "
            f"(detected: {max_pins_detected}, total tracked: {len(pin_tracker.get_pins())})"
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
        if display_enabled:
            cv.destroyAllWindows()
