from __future__ import annotations

from typing import Sequence

import cv2 as cv
import numpy as np


def _as_int_bbox(bbox: Sequence[int] | Sequence[float] | None) -> tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = [int(v) for v in bbox]
    return x, y, w, h


def annotate_bbox(
    frame_bgr: np.ndarray,
    bbox: Sequence[int] | Sequence[float] | None,
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    int_bbox = _as_int_bbox(bbox)
    if int_bbox is None:
        return frame_bgr.copy()

    x, y, w, h = int_bbox
    annotated = frame_bgr.copy()
    cv.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

    (text_w, text_h), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = max(0, x)
    text_y = max(text_h + 6, y - 8)
    cv.rectangle(
        annotated,
        (text_x, text_y - text_h - baseline - 4),
        (text_x + text_w + 6, text_y + baseline),
        color,
        thickness=-1,
    )
    cv.putText(
        annotated,
        label,
        (text_x + 3, text_y - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv.LINE_AA,
    )
    return annotated


def draw_fading_trajectory(
    frame_bgr: np.ndarray,
    trajectory_points: Sequence[Sequence[int]],
    color: tuple[int, int, int] = (0, 255, 255),
    max_points: int = 200,
    min_alpha: float = 0.25,
) -> np.ndarray:
    if not trajectory_points:
        return frame_bgr.copy()

    points = np.asarray(trajectory_points[-max_points:], dtype=np.int32)
    if points.ndim != 2 or points.shape[0] < 1:
        return frame_bgr.copy()

    min_alpha = float(np.clip(min_alpha, 0.0, 1.0))
    output = frame_bgr.copy()

    if points.shape[0] >= 2:
        segment_count = points.shape[0] - 1
        white = np.array([255, 255, 255], dtype=np.float32)
        base = np.array(color, dtype=np.float32)

        for idx in range(1, points.shape[0]):
            # Older segments look lighter by blending toward white.
            age = 1.0 - (idx / segment_count)
            blend = (1.0 - min_alpha) * age
            segment_color = (base * (1.0 - blend) + white * blend).astype(np.int32)
            p0 = tuple(int(v) for v in points[idx - 1])
            p1 = tuple(int(v) for v in points[idx])
            cv.line(output, p0, p1, tuple(int(v) for v in segment_color), 3, cv.LINE_AA)

    for point in points:
        cv.circle(output, (int(point[0]), int(point[1])), 3, color, -1)

    return output


def annotate_tracking_frame(
    frame_bgr: np.ndarray,
    bbox: Sequence[int] | Sequence[float] | None,
    label: str,
    trajectory_points: Sequence[Sequence[int]],
    box_color: tuple[int, int, int] = (0, 255, 0),
    trajectory_color: tuple[int, int, int] = (0, 255, 255),
    trail_length: int = 200,
    trail_min_alpha: float = 0.25,
) -> np.ndarray:
    with_box = annotate_bbox(frame_bgr, bbox, label, box_color)
    return draw_fading_trajectory(with_box, trajectory_points, trajectory_color, trail_length, trail_min_alpha)

