"""
annotation_renderer.py — Composable frame annotation utilities.

Functions
---------
annotate_bbox            Draw a single labelled bounding box.
draw_fading_trajectory   Smooth, gradient, fading car trail.
draw_pin_annotations     Draw all pin states (standing / fallen).
draw_score_overlay       Draw the score card (elapsed time + pins hit).
annotate_tracking_frame  Compose all of the above into one annotated frame.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2 as cv
import numpy as np


DEFAULT_TRAIL_LENGTH: int = 200
DEFAULT_TRAIL_MIN_ALPHA: float = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory smoothing — Catmull-Rom spline
# ─────────────────────────────────────────────────────────────────────────────

def _catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Evaluate one Catmull-Rom segment between p1 and p2."""
    t  = np.linspace(0.0, 1.0, num_samples, endpoint=False)
    t2 = t * t
    t3 = t2 * t
    # Standard Catmull-Rom matrix (tension=0.5)
    x = 0.5 * (
        2 * p1[0]
        + (-p0[0] + p2[0])             * t
        + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
        + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])  * t3
    )
    y = 0.5 * (
        2 * p1[1]
        + (-p0[1] + p2[1])             * t
        + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
        + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])  * t3
    )
    return np.stack([x, y], axis=1)


def _smooth_trajectory(
    points: np.ndarray,
    samples_per_segment: int = 8,
) -> np.ndarray:
    """
    Return a smooth curve through all control points using Catmull-Rom.

    Works for any number of points >= 2.  For 2-3 points it degrades
    gracefully to a straight line.
    """
    n = len(points)
    if n < 2:
        return points

    # Pad start and end so every point is an interior knot
    pts = np.vstack([points[:1], points, points[-1:]])   # shape (n+2, 2)
    segments = []
    for i in range(1, len(pts) - 2):
        seg = _catmull_rom_segment(
            pts[i - 1].astype(float),
            pts[i].astype(float),
            pts[i + 1].astype(float),
            pts[i + 2].astype(float),
            samples_per_segment,
        )
        segments.append(seg)

    # Append the very last point
    smooth = np.vstack(segments + [points[-1:]])
    return smooth.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_color(
    t: float,                          # 0.0 = oldest (tail) → 1.0 = newest (head)
    tail_hsv: Tuple[int, int, int] = (135, 200, 160),   # purple-ish
    head_hsv: Tuple[int, int, int] = (80,  255, 255),   # bright cyan-green
) -> Tuple[int, int, int]:
    """
    Interpolate between two HSV colours and return BGR.

    OpenCV HSV: H∈[0,179], S∈[0,255], V∈[0,255].
    Default: purple (old) → cyan-green (new).
    """
    t = float(np.clip(t, 0.0, 1.0))
    h = int(round(tail_hsv[0] + (head_hsv[0] - tail_hsv[0]) * t))
    s = int(round(tail_hsv[1] + (head_hsv[1] - tail_hsv[1]) * t))
    v = int(round(tail_hsv[2] + (head_hsv[2] - tail_hsv[2]) * t))
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


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
        color, thickness=-1,
    )
    cv.putText(
        annotated, label,
        (text_x + 3, text_y - 4),
        font, 0.7, (0, 0, 0), 2, cv.LINE_AA,
    )
    return annotated


def draw_fading_trajectory(
    frame_bgr: np.ndarray,
    trajectory_points: Sequence[Sequence[int]],
    color: Tuple[int, int, int] = (0, 255, 255),   # kept for API compat; ignored
    max_points: int = DEFAULT_TRAIL_LENGTH,
    min_alpha: float = DEFAULT_TRAIL_MIN_ALPHA,
) -> np.ndarray:
    """
    Draw the car's path as a smooth, gradient, fading trail.

    Improvements over the original
    -------------------------------
    * **Smooth** — Catmull-Rom spline interpolation between the raw GPS-like
      control points eliminates the jagged polyline appearance.
    * **Gradient** — colour transitions from purple (oldest) through blue to
      bright cyan-green (most recent) along the length of the trail.
    * **Fade** — segments age toward near-transparency; only the leading tip is
      fully opaque.  Implemented with per-segment ``addWeighted`` blending.
    * **Taper** — line thickness decreases from head (3 px) to tail (1 px),
      reinforcing the sense of recency.

    The ``color`` parameter is kept for backward-compatibility but is now
    ignored in favour of the built-in HSV gradient.
    """
    if not trajectory_points:
        return frame_bgr.copy()

    raw = np.asarray(trajectory_points[-max_points:], dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] < 1:
        return frame_bgr.copy()

    if raw.shape[0] == 1:
        out = frame_bgr.copy()
        cv.circle(out, tuple(raw[0]), 4, _gradient_color(1.0), -1)
        return out

    # ── Smooth via Catmull-Rom ────────────────────────────────────────────
    smooth = _smooth_trajectory(raw, samples_per_segment=8)   # shape (M, 2)
    M = len(smooth)
    if M < 2:
        return frame_bgr.copy()

    output = frame_bgr.copy()
    min_alpha = float(np.clip(min_alpha, 0.0, 1.0))

    # ── Draw each micro-segment with gradient colour + alpha fade ─────────
    # We blend into a scratch canvas per-segment to get true per-segment alpha.
    # This is O(M) addWeighted calls but M ≈ trail_length * samples_per_segment
    # and addWeighted on a typical 1080p frame takes ~0.3 ms — acceptable.
    for i in range(1, M):
        # t ∈ [0,1]: 0 = tail (oldest), 1 = head (newest)
        t         = i / (M - 1)
        seg_color = _gradient_color(t)

        # Alpha: linear from min_alpha at tail to 1.0 at head
        alpha = min_alpha + (1.0 - min_alpha) * t

        # Thickness: 1 px at tail → 3 px at head
        thickness = max(1, int(round(1 + 2 * t)))

        p0 = tuple(int(v) for v in smooth[i - 1])
        p1 = tuple(int(v) for v in smooth[i])

        if alpha >= 0.99:
            # Fully opaque — draw directly (fast path)
            cv.line(output, p0, p1, seg_color, thickness, cv.LINE_AA)
        else:
            scratch = output.copy()
            cv.line(scratch, p0, p1, seg_color, thickness, cv.LINE_AA)
            cv.addWeighted(scratch, alpha, output, 1.0 - alpha, 0.0, output)

    # ── Bright dot at the current tip ────────────────────────────────────
    head_color = _gradient_color(1.0)
    cv.circle(output, tuple(int(v) for v in smooth[-1]), 5, head_color, -1, cv.LINE_AA)
    cv.circle(output, tuple(int(v) for v in smooth[-1]), 7, head_color, 1,  cv.LINE_AA)

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Pin annotation
# ─────────────────────────────────────────────────────────────────────────────

def draw_pin_annotations(
    frame_bgr: np.ndarray,
    pin_states: list,
    show_roi: bool = False,
) -> np.ndarray:
    """
    Draw all pin bounding boxes onto ``frame_bgr``.

    Standing pins are shown as plain unlabeled boxes.
    Fallen pins show only their crash order number, with no extra badge box.
    """
    if not pin_states:
        return frame_bgr.copy()

    out = frame_bgr.copy()

    for pin in pin_states:
        x, y, w, h = [int(v) for v in pin.bbox]

        if pin.is_fallen:
            grey = (130, 130, 130)
            cv.rectangle(out, (x, y), (x + w, y + h), grey, 2)
            cv.line(out, (x, y), (x + w, y + h), grey, 2)
            cv.line(out, (x + w, y), (x, y + h), grey, 2)

            text  = str(pin.fall_order)
            font  = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.9
            thick = 2
            (tw, th), baseline = cv.getTextSize(text, font, scale, thick)
            cx = x + w // 2
            cy = y + h // 2
            tx = max(0, cx - tw // 2)
            ty = max(th + 2, cy + th // 2)
            cv.putText(out, text, (tx, ty), font, scale, (255, 255, 255), thick, cv.LINE_AA)

        else:
            color = (255, 255, 255)
            cv.rectangle(out, (x, y), (x + w, y + h), color, 2)

        if show_roi:
            # Intentionally ignored: no secondary ROI boxes are drawn.
            pass

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Score overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_score_overlay(
    frame_bgr: np.ndarray,
    elapsed_seconds: float,
    pins_fallen: int,
    pin_states: Optional[list] = None,
    position: Tuple[int, int] = (20, 20),
) -> np.ndarray:
    """Semi-transparent score card in the top-left corner."""
    out     = frame_bgr.copy()
    overlay = out.copy()

    lines = [
        f"Time:  {elapsed_seconds:.1f}s",
        f"Pins:  {pins_fallen} knocked down",
    ]

    if not lines:
        return out

    font   = cv.FONT_HERSHEY_SIMPLEX
    scale  = 0.65
    thick  = 1
    pad    = 10
    line_h = 26

    max_w   = max(cv.getTextSize(ln, font, scale, thick)[0][0] for ln in lines)
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
    trajectory_color: Tuple[int, int, int] = (0, 255, 255),  # kept for compat
    trail_length: int = DEFAULT_TRAIL_LENGTH,
    trail_min_alpha: float = DEFAULT_TRAIL_MIN_ALPHA,
    pin_states: Optional[list] = None,
    elapsed_seconds: Optional[float] = None,
    show_roi: bool = False,
) -> np.ndarray:
    """
    Compose all annotations into one frame.

    Layer order (bottom → top):
      1. Smooth gradient fading car trajectory
      2. Car bounding box + label
      3. Pin boxes / fall-order badges
      4. Score overlay
    """
    # ── CRITICAL: Force label to "car" to prevent class mixup ──────────────
    safe_label = "car"
    out = draw_fading_trajectory(
        frame_bgr, trajectory_points, trajectory_color, trail_length, trail_min_alpha
    )
    out = annotate_bbox(out, bbox, safe_label, box_color)
    if pin_states is not None:
        out = draw_pin_annotations(out, pin_states, show_roi=show_roi)
    if elapsed_seconds is not None and pin_states is not None:
        fallen = sum(1 for p in pin_states if p.is_fallen)
        out = draw_score_overlay(out, elapsed_seconds, fallen, pin_states)
    return out