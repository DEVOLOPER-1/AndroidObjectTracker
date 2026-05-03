from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm.auto import tqdm

from abavit_tracker import AbaViTracker
from annotation_renderer import annotate_tracking_frame
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


def _pick_initial_detection(detections, preferred_class: str):
    if not detections:
        return None

    preferred = [det for det in detections if getattr(det, "object_type", None) == preferred_class]
    pool = preferred if preferred else detections
    return max(pool, key=lambda det: float(getattr(det, "confidence", 0.0)))


def _find_initial_detection(cap, yolo: YOLOTracker, preferred_class: str, max_search_frames: int):
    for frame_index in range(max_search_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        detections = yolo.infer(frame_bgr, is_bgr=True)
        detection = _pick_initial_detection(detections, preferred_class=preferred_class)
        if detection is not None:
            return frame_index, frame_bgr, detection

    return None, None, None


def _center_from_bbox(bbox: list[int] | tuple[int, int, int, int]) -> list[int]:
    x, y, w, h = [int(v) for v in bbox]
    return [x + w // 2, y + h // 2]


def run_pipeline(config: PipelineConfig) -> Path:
    process_every_n = max(1, int(config.process_every_n))
    max_search_frames = max(1, int(config.max_search_frames))
    trail_length = max(2, int(config.trail_length))

    yolo = YOLOTracker()
    params = type("P", (), {})()
    params.tracker_name = "abavit"
    params.param_name = "default"
    abavit = AbaViTracker(params, dataset_name="custom")

    cap = cv.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {config.video_path}")

    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    first_frame_index, first_frame_bgr, detection = _find_initial_detection(
        cap,
        yolo,
        preferred_class=config.preferred_class,
        max_search_frames=max_search_frames,
    )
    if first_frame_bgr is None or detection is None:
        cap.release()
        raise SystemExit("No detections from YOLO to initialize AbaViTracker")

    x1, y1, x2, y2 = detection.box
    init_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
    last_valid_bbox = init_bbox
    label = getattr(detection, "object_type", "object")
    trajectory_points = [_center_from_bbox(init_bbox)]

    first_frame_rgb = cv.cvtColor(np.asarray(first_frame_bgr), cv.COLOR_BGR2RGB)
    abavit.initialize(first_frame_rgb, {"init_bbox": init_bbox})

    output_path = config.output_path
    if output_path is None:
        output_path = config.video_path.with_name(f"{config.video_path.stem}_tracked.mp4")

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or first_frame_bgr.shape[1])
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or first_frame_bgr.shape[0])
    writer = cv.VideoWriter(
        str(output_path),
        getattr(cv, "VideoWriter_fourcc")(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open output video for writing: {output_path}")

    window_name = "AbaViTrack"
    if config.display:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    progress_total = total_frames if total_frames > 0 else None
    pbar = tqdm(total=progress_total, desc=f"Tracking {config.video_path.name}", unit="frame")

    try:
        if first_frame_index is not None:
            pbar.update(first_frame_index + 1)

        first_annotated = annotate_tracking_frame(
            np.asarray(first_frame_bgr),
            last_valid_bbox,
            label,
            trajectory_points,
            trail_length=trail_length,
            trail_min_alpha=config.trail_min_alpha,
        )
        writer.write(first_annotated)

        if config.display:
            cv.imshow(window_name, first_annotated)
            if cv.waitKey(1) & 0xFF == ord("q"):
                return output_path

        frame_index = (first_frame_index or 0) + 1
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            should_process = ((frame_index - (first_frame_index or 0)) % process_every_n) == 0
            if should_process:
                frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                out = abavit.track(frame_rgb)
                tracked_bbox = out.get("target_bbox") if isinstance(out, dict) else None
                if isinstance(tracked_bbox, (list, tuple)) and len(tracked_bbox) == 4:
                    last_valid_bbox = [int(v) for v in tracked_bbox]
                    trajectory_points.append(_center_from_bbox(last_valid_bbox))

            annotated = annotate_tracking_frame(
                frame_bgr,
                last_valid_bbox,
                label,
                trajectory_points,
                trail_length=trail_length,
                trail_min_alpha=config.trail_min_alpha,
            )
            writer.write(annotated)
            pbar.update(1)

            if config.display:
                cv.imshow(window_name, annotated)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1

        return output_path
    finally:
        cap.release()
        writer.release()
        pbar.close()
        if config.display:
            cv.destroyAllWindows()


