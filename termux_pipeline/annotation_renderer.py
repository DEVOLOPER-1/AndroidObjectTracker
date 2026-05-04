"""
annotation_renderer.py — Composable frame annotation utilities.

Functions
---------
annotate_bbox            Draw a single labelled bounding box.
draw_fading_trajectory   Draw the car's fading colour trail.
draw_pin_annotations     Draw all pin states (standing / fallen).
draw_score_overlay       Draw the final score card (elapsed time + pins hit).
annotate_tracking_frame  Compose all of the above into one annotated frame.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


# -------------------------
# Tweakable display defaults
# -------------------------
DEFAULT_TRAIL_LENGTH: int = 200
DEFAULT_TRAIL_MIN_ALPHA: float = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Car tracker annotation
# ─────────────────────────────────────────────────────────────────────────────

def _as_int_bbox(
    bbox: Sequence[int] | Sequence[float] | None,
) -> Tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = [int(v) for v in bbox]
    return x, y, w, h


def annotate_bbox(
    frame_bgr: np.ndarray,
    bbox: Sequence[int] | Sequence[float] | None,
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a single labelled bounding box.  Returns a copy."""
    int_bbox = _as_int_bbox(bbox)
    if int_bbox is None:
        return frame_bgr.copy()

    x, y, w, h = int_bbox
    annotated = frame_bgr.copy()
    cv.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

    font = cv.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv.getTextSize(label, font, 0.7, 2)
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
        annotated, label,
        (text_x + 3, text_y - 4),
        font, 0.7,
        (0, 0, 0),
        2, cv.LINE_AA,
    )
    return annotated


def draw_fading_trajectory(
    frame_bgr: np.ndarray,
    trajectory_points: Sequence[Sequence[int]],
    color: Tuple[int, int, int] = (0, 255, 255),
    max_points: int = DEFAULT_TRAIL_LENGTH,
    min_alpha: float = DEFAULT_TRAIL_MIN_ALPHA,
) -> np.ndarray:
    """
    Draw the car's path as a fading line.

    Older segments are blended toward white so the most recent position
    is brightest.  Returns a copy of the frame.
    """
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
        base  = np.array(color, dtype=np.float32)

        for idx in range(1, points.shape[0]):
            age   = 1.0 - (idx / segment_count)
            blend = (1.0 - min_alpha) * age
            seg_color = (base * (1.0 - blend) + white * blend).astype(np.int32)
            p0 = tuple(int(v) for v in points[idx - 1])
            p1 = tuple(int(v) for v in points[idx])
            cv.line(output, p0, p1, tuple(int(v) for v in seg_color), 3, cv.LINE_AA)

    for point in points:
        cv.circle(output, (int(point[0]), int(point[1])), 3, color, -1)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Pin annotation  (imported lazily to avoid circular dependency)
# ─────────────────────────────────────────────────────────────────────────────

def draw_pin_annotations(
    frame_bgr: np.ndarray,
    pin_states: list,          # List[pin_tracker.PinState]  (typed loosely to avoid import)
    show_roi: bool = False,
) -> np.ndarray:
    """
    Draw all pin bounding boxes and labels onto `frame_bgr`.

    Standing pins : coloured rectangle + colour name label above the box.
    Fallen  pins  : grey crossed-out box + blue fall-order badge centred on
                    the original pin location.  The badge persists for the
                    rest of the video (is_fallen stays True).

    Parameters
    ----------
    frame_bgr  : H×W×3 uint8 BGR image.
    pin_states : List of PinState objects returned by PinTracker.update().
    show_roi   : Draw the fall-detection ROI as a thin cyan box (debug).

    Returns
    -------
    Annotated BGR image (copy).
    """
    if not pin_states:
        return frame_bgr.copy()

    out = frame_bgr.copy()

    for pin in pin_states:
        x, y, w, h = [int(v) for v in pin.bbox]

        if pin.is_fallen:
            # ── Fallen: grey crossed-out box + fall-order badge ────────────
            grey = (110, 110, 110)
            cv.rectangle(out, (x, y), (x + w, y + h), grey, 2)
            cv.line(out, (x, y), (x + w, y + h), grey, 2)
            cv.line(out, (x + w, y), (x, y + h), grey, 2)

            # Large centred badge with the fall-order number
            text     = f"#{pin.fall_order}"
            font     = cv.FONT_HERSHEY_SIMPLEX
            scale    = 0.95
            thick    = 2
            (tw, th), bl = cv.getTextSize(text, font, scale, thick)
            cx = x + w // 2
            cy = y + h // 2
            pad = 6
            cv.rectangle(out,
                          (cx - tw//2 - pad, cy - th//2 - pad),
                          (cx + tw//2 + pad, cy + th//2 + pad + bl),
                          (0, 0, 180), -1)
            cv.rectangle(out,
                          (cx - tw//2 - pad, cy - th//2 - pad),
                          (cx + tw//2 + pad, cy + th//2 + pad + bl),
                          (255, 255, 255), 1)
            cv.putText(out, text,
                       (cx - tw // 2, cy + th // 2),
                       font, scale, (255, 255, 255), thick, cv.LINE_AA)

        else:
            # ── Standing: coloured box + colour-name label ─────────────────
            color = tuple(int(v) for v in pin.draw_bgr)
            cv.rectangle(out, (x, y), (x + w, y + h), color, 3)

            label = pin.color_name
            font  = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 1
            (lw, lh), baseline = cv.getTextSize(label, font, scale, thick)
            lx = max(0, x)
            ly = max(lh + 6, y - 6)
            text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
            cv.rectangle(out,
                          (lx, ly - lh - 4),
                          (lx + lw + 6, ly + baseline + 2),
                          color, -1)
            cv.putText(out, label, (lx + 3, ly - 2),
                       font, scale, text_color, thick, cv.LINE_AA)

        # ── Debug: padded ROI ──────────────────────────────────────────────
        if show_roi and hasattr(pin, "roi"):
            rx, ry, rw, rh = [int(v) for v in pin.roi]
            cv.rectangle(out, (rx, ry), (rx + rw, ry + rh), (180, 180, 60), 1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Score overlay  (shown in the final frame / RESULTS screen)
# ─────────────────────────────────────────────────────────────────────────────

def draw_score_overlay(
    frame_bgr: np.ndarray,
    elapsed_seconds: float,
    pins_fallen: int,
    pin_states: Optional[list] = None,
    position: Tuple[int, int] = (20, 20),
) -> np.ndarray:
    """
    Draw a semi-transparent score card in the top-left corner.

    Shows elapsed time, total pins knocked down, and (if pin_states is
    provided) a one-line fall-order summary.

    Returns a copy.
    """
    out = frame_bgr.copy()
    overlay = out.copy()

    lines = [
        f"Time:  {elapsed_seconds:.1f}s",
        f"Pins:  {pins_fallen} knocked down",
    ]

    if pin_states:
        fallen = sorted(
            [p for p in pin_states if p.is_fallen],
            key=lambda p: p.fall_order,
        )
        if fallen:
            seq = " -> ".join(f"#{p.fall_order}({p.color_name})" for p in fallen)
            lines.append(f"Order: {seq}")

    font  = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thick = 1
    pad   = 10
    line_h = 26

    # Measure total overlay size
    max_w = max(cv.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    total_h = len(lines) * line_h + 2 * pad
    total_w = max_w + 2 * pad

    x0, y0 = position
    cv.rectangle(overlay, (x0, y0), (x0 + total_w, y0 + total_h), (20, 20, 20), -1)
    cv.addWeighted(overlay, 0.60, out, 0.40, 0, out)

    for i, line in enumerate(lines):
        cv.putText(
            out, line,
            (x0 + pad, y0 + pad + (i + 1) * line_h - 4),
            font, scale, (220, 220, 220), thick, cv.LINE_AA,
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Master compositor
# ─────────────────────────────────────────────────────────────────────────────

def annotate_tracking_frame(
    frame_bgr: np.ndarray,
    bbox: Sequence[int] | Sequence[float] | None,
    label: str,
    trajectory_points: Sequence[Sequence[int]],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    trajectory_color: Tuple[int, int, int] = (0, 255, 255),
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    trail_min_alpha: float = DEFAULT_TRAIL_MIN_ALPHA,
    pin_states: Optional[list] = None,
    elapsed_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Compose all annotations into one frame.

    Layer order (bottom to top):
      1. Fading car trajectory
      2. Car bounding box + label
      3. Pin bounding boxes + labels / fall-order badges
      4. Score overlay (only when elapsed_seconds is provided)

    Parameters
    ----------
    frame_bgr         : Source BGR frame (not modified).
    bbox              : Car bbox [x, y, w, h] or None if lost.
    label             : Car label string (e.g. "car").
    trajectory_points : List of [cx, cy] centre points for the trail.
    box_color         : Car bounding-box colour.
    trajectory_color  : Trail colour (older segments fade toward white).
    trail_length      : Maximum trajectory points to draw.
    trail_min_alpha   : Oldest segment intensity (0.0 = white, 1.0 = full colour).
    pin_states        : List[PinState] from PinTracker.update(); None → no pins drawn.
    elapsed_seconds   : If provided, draws the score overlay card.

    Returns
    -------
    Annotated BGR frame (copy).
    """
    # Layer 1: trajectory
    out = draw_fading_trajectory(
        frame_bgr, trajectory_points, trajectory_color, trail_length, trail_min_alpha
    )
    # Layer 2: car box
    out = annotate_bbox(out, bbox, label, box_color)
    # Layer 3: pins
    if pin_states is not None:
        out = draw_pin_annotations(out, pin_states)
    # Layer 4: score card
    if elapsed_seconds is not None and pin_states is not None:
        fallen = sum(1 for p in pin_states if p.is_fallen)
        out = draw_score_overlay(out, elapsed_seconds, fallen, pin_states)

    return out
