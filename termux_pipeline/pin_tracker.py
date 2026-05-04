"""
pin_tracker.py — Colour-based bowling-pin detection and fall-order tracker.

Architecture
------------
Frame 0  →  PinTracker.initialize(frame_rgb)
                Detects all upright pins by HSV colour segmentation + shape
                filter.  Saves each pin's bounding box and a baseline
                colour-pixel count inside a padded ROI.

Frame N  →  PinTracker.update(frame_rgb)
                For every still-standing pin, counts how many of its defining
                colour pixels remain inside its fixed ROI.  If that count drops
                below FALL_COLOR_RATIO × baseline for FALL_CONFIRM_N
                consecutive frames the pin is marked fallen and assigned the
                next fall-order integer (1, 2, 3 …).

Why colour + shape and NOT YOLO?
---------------------------------
The YOLO model in this codebase was trained on COCO classes and labels the
bowling pins as "toothbrush", "bird", "scissors", etc.  Colour segmentation
is deterministic, runs on CPU in microseconds, and is not affected by domain
mismatch.

Colour ranges (HSV, OpenCV scale H∈[0,180])
--------------------------------------------
Calibrated directly from the target scene image:
  green  H ≈ 49-53, S > 100
  blue   H ≈ 108,   S > 150
  red    H ≈ 176,   S > 150  (upper-hue red segment only, sufficient here)
  yellow H ≈ 19,    S > 180

Note on the RC car
-------------------
The user's RC car is white (S ≈ 0-40 in HSV).  None of the colour ranges
below require high saturation from a white source, so the car generates
zero false detections.  If a coloured car is ever used, add its bounding
box to the `exclude_regions` parameter of `initialize()`.
"""

from __future__ import annotations

import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# HSV colour ranges (OpenCV: H∈[0,180], S∈[0,255], V∈[0,255])
# Each colour maps to a list of (lower, upper) HSV bound pairs.
# Red wraps around the hue circle so it can have two ranges;
# for this scene only the upper segment (H≈155-180) is needed.
# ─────────────────────────────────────────────────────────────────────────────
_HSV_RANGES: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = {
    "green":  [((35,  100,  50), ( 70, 255, 255))],
    "blue":   [((90,  150,  60), (125, 255, 255))],
    "red":    [
        ((  0, 150,  60), ( 12, 255, 255)),   # lower-hue red
        ((155, 150,  60), (180, 255, 255)),   # upper-hue red (dominant here H≈176)
    ],
    "yellow": [((10,  180, 150), ( 30, 255, 255))],
    # Extend here if your pin set includes additional colours:
    # "orange": [((10, 150, 100), (20, 255, 255))],
    # "purple": [((125, 60, 50), (160, 255, 255))],
}

# BGR (OpenCV) drawing colour per pin colour
_BGR: Dict[str, Tuple[int, int, int]] = {
    "green":  (  0, 210,   0),
    "blue":   (255,  50,  50),
    "red":    (  0,   0, 230),
    "yellow": (  0, 210, 255),
    "orange": (  0, 140, 255),
    "purple": (200,   0, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# Detection / tracking parameters
# ─────────────────────────────────────────────────────────────────────────────
_MIN_BLOB_AREA   = 1200    # px² — blobs smaller than this are noise / car glints
_MIN_ASPECT      = 1.20   # h/w — upright pins are taller than wide
_MAX_ASPECT      = 8.00   # sanity cap
_NMS_IOU_THRESH  = 0.25   # merge overlapping detections across colours
_ROI_PAD         = 0.45   # expand detection bbox by this fraction each side
                           #   for the fall-check ROI
_FALL_COLOR_RATIO  = 0.38 # colour pixels must drop below this × baseline
_FALL_CONFIRM_N    = 4    # consecutive frames below threshold → confirmed fall
_MORPH_OPEN_K    = 5      # morphological open kernel (noise removal)
_MORPH_CLOSE_K   = 13     # morphological close kernel (fills pin body gaps)


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PinState:
    """All state associated with one detected bowling pin."""
    pin_id:      int
    color_name:  str
    draw_bgr:    Tuple[int, int, int]   # BGR colour for OpenCV drawing
    bbox:        List[int]              # [x, y, w, h] tight box (fixed after init)
    center:      Tuple[int, int]        # (cx, cy) of the bbox
    roi:         List[int]              # [x, y, w, h] padded ROI for fall detection
    baseline_px: int                    # colour pixels in roi at frame 0

    is_fallen:   bool = False
    fall_order:  int  = 0               # 0 → standing;  1, 2, … → fallen in order

    # internal fall-confirmation counter (not part of the public API)
    _candidate_n: int = field(default=0, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class PinTracker:
    """
    Detect and track bowling-pin falls using HSV colour segmentation.

    Example
    -------
    tracker = PinTracker()
    tracker.initialize(first_frame_rgb)        # Frame 0

    for frame_rgb in subsequent_frames:
        pin_states = tracker.update(frame_rgb) # Frame 1+
        annotated  = tracker.draw(frame_bgr)   # optional, returns annotated copy
    """

    def __init__(self) -> None:
        self.pins: List[PinState] = []
        self._fall_counter: int = 0    # increments each time a pin falls

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(
        self,
        frame_rgb: np.ndarray,
        exclude_regions: Optional[List[List[int]]] = None,
    ) -> List[PinState]:
        """
        Detect all upright bowling pins in the first frame.

        Parameters
        ----------
        frame_rgb:
            H×W×3 uint8 RGB image.
        exclude_regions:
            Optional list of [x, y, w, h] boxes to black out before detection
            (use this if the car has a colour that overlaps with pin colours).

        Returns
        -------
        List of PinState objects, one per detected pin.
        """
        self.pins = []
        self._fall_counter = 0

        # Optionally mask out known non-pin regions (e.g. coloured car body)
        frame_work = frame_rgb.copy()
        if exclude_regions:
            for rx, ry, rw, rh in exclude_regions:
                frame_work[ry:ry+rh, rx:rx+rw] = 128  # neutral grey

        hsv = cv.cvtColor(frame_work, cv.COLOR_RGB2HSV)
        # Mild blur before segmentation to reduce specular hotspots on the pins
        hsv_blurred = cv.GaussianBlur(hsv, (5, 5), 1.2)

        # ── Detect blobs per colour ──────────────────────────────────────────
        candidates: List[Tuple[List[int], int, str]] = []  # (bbox, area, color)
        for color_name in _HSV_RANGES:
            mask = _make_color_mask(hsv_blurred, color_name)
            for bbox, area in _extract_pin_blobs(mask):
                candidates.append((list(bbox), area, color_name))

        # ── NMS across colours ───────────────────────────────────────────────
        candidates = _nms(candidates, _NMS_IOU_THRESH)

        # ── Build PinState objects ───────────────────────────────────────────
        H, W = frame_rgb.shape[:2]
        for idx, (bbox, area, color_name) in enumerate(candidates):
            x, y, w, h = bbox
            cx = x + w // 2
            cy = y + h // 2
            roi = _padded_roi(bbox, W, H, _ROI_PAD)

            # Baseline: count how many colour pixels are in the padded ROI now
            baseline = _count_color_pixels(hsv_blurred, color_name, roi)
            baseline = max(baseline, 1)  # guard against zero division

            pin = PinState(
                pin_id      = idx,
                color_name  = color_name,
                draw_bgr    = _BGR.get(color_name, (200, 200, 200)),
                bbox        = [x, y, w, h],
                center      = (cx, cy),
                roi         = roi,
                baseline_px = baseline,
            )
            self.pins.append(pin)

        print(
            f"[PinTracker] Initialised {len(self.pins)} pins: "
            + ", ".join(f"{p.color_name}@({p.center[0]},{p.center[1]})" for p in self.pins)
        )
        return list(self.pins)

    def update(self, frame_rgb: np.ndarray) -> List[PinState]:
        """
        Evaluate which pins have fallen in the current frame.

        Called every frame after initialize().  Cheap: only counts pixels in
        small fixed ROIs — no model inference needed.

        Parameters
        ----------
        frame_rgb : H×W×3 uint8 RGB image.

        Returns
        -------
        Snapshot list of all PinState objects (modifications applied in-place).
        """
        if not self.pins:
            return []

        hsv = cv.cvtColor(frame_rgb, cv.COLOR_RGB2HSV)
        hsv_blurred = cv.GaussianBlur(hsv, (5, 5), 1.2)

        for pin in self.pins:
            if pin.is_fallen:
                continue  # already confirmed — nothing more to do

            current_px = _count_color_pixels(hsv_blurred, pin.color_name, pin.roi)
            ratio = current_px / pin.baseline_px

            if ratio < _FALL_COLOR_RATIO:
                pin._candidate_n += 1
                # Debug: uncomment to see live ratio readings
                # print(f"[PinTracker] Pin {pin.pin_id} ({pin.color_name}) ratio={ratio:.2f}  n={pin._candidate_n}")
            else:
                # Evidence of standing — decay the counter (forgives brief occlusions
                # by the car passing over the pin without actually knocking it)
                pin._candidate_n = max(0, pin._candidate_n - 1)

            if pin._candidate_n >= _FALL_CONFIRM_N:
                self._fall_counter += 1
                pin.is_fallen   = True
                pin.fall_order  = self._fall_counter
                print(
                    f"[PinTracker] *** PIN FELL ***  id={pin.pin_id} "
                    f"color={pin.color_name}  fall_order=#{pin.fall_order}  "
                    f"ratio={ratio:.2f}"
                )

        return list(self.pins)

    def get_pins(self) -> List[PinState]:
        """Return a snapshot of the current pin states (thread-safe copy)."""
        return list(self.pins)

    def fallen_count(self) -> int:
        """Number of pins confirmed fallen so far."""
        return self._fall_counter

    def draw(
        self,
        frame_bgr: np.ndarray,
        show_roi: bool = False,
    ) -> np.ndarray:
        """
        Draw pin annotations onto a BGR frame.

        Standing pins: coloured bounding box + colour label.
        Fallen  pins : greyed-out crossed box + prominent fall-order number.

        Parameters
        ----------
        frame_bgr : H×W×3 uint8 BGR image (modified copy returned).
        show_roi  : If True, also draws the padded fall-detection ROI as a
                    thin rectangle (useful for debugging thresholds).

        Returns
        -------
        Annotated BGR image.
        """
        out = frame_bgr.copy()
        for pin in self.pins:
            x, y, w, h = pin.bbox

            if pin.is_fallen:
                # ── Fallen: grey cross + fall-order badge ──────────────────
                grey = (110, 110, 110)
                cv.rectangle(out, (x, y), (x+w, y+h), grey, 2)
                cv.line(out, (x, y), (x+w, y+h), grey, 2)
                cv.line(out, (x+w, y), (x, y+h), grey, 2)
                # Big numbered badge centred on the original pin location
                _draw_badge(out, f"#{pin.fall_order}", x + w//2, y + h//2,
                            text_color=(255, 255, 255), bg_color=(0, 0, 180),
                            scale=1.0, thickness=2)
            else:
                # ── Standing: coloured box + colour-name label ─────────────
                color = pin.draw_bgr
                cv.rectangle(out, (x, y), (x+w, y+h), color, 3)
                _draw_label(out, pin.color_name, x, y - 6, color)

            if show_roi:
                rx, ry, rw, rh = pin.roi
                cv.rectangle(out, (rx, ry), (rx+rw, ry+rh), (180, 180, 60), 1)

        return out

    def reset(self) -> None:
        """Clear all state (call before reprocessing a video)."""
        self.pins = []
        self._fall_counter = 0


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

# Pre-allocate morphological kernels once
_KERNEL_OPEN  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_MORPH_OPEN_K,  _MORPH_OPEN_K))
_KERNEL_CLOSE = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_MORPH_CLOSE_K, _MORPH_CLOSE_K))


def _make_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """Binary mask: pixels belonging to `color_name` after morphological cleanup."""
    ranges = _HSV_RANGES[color_name]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    # Open: removes isolated specks and thin noise
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  _KERNEL_OPEN)
    # Close: fills gaps in the pin body caused by specular highlights
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, _KERNEL_CLOSE)
    return mask


def _extract_pin_blobs(
    mask: np.ndarray,
) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """
    Find contours in `mask` that pass the area + aspect-ratio filters.

    Returns a list of (bbox_xywh, area) for valid pin candidates.
    """
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = int(cv.contourArea(cnt))
        if area < _MIN_BLOB_AREA:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        if w < 1:
            continue
        aspect = h / float(w)
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            # Not tall enough to be an upright pin (or absurdly thin → noise)
            continue
        results.append(((x, y, w, h), area))
    return results


def _iou(
    b1: Sequence[int],
    b2: Sequence[int],
) -> float:
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def _nms(
    candidates: List[Tuple[List[int], int, str]],
    iou_thresh: float,
) -> List[Tuple[List[int], int, str]]:
    """
    Non-maximum suppression across all colour detections.

    When two candidates overlap (IoU > iou_thresh), keep the larger one.
    This prevents double-counting a pin that partially falls inside two
    colour ranges (e.g. an orange pin matching both "red" and "yellow").
    """
    candidates = sorted(candidates, key=lambda c: -c[1])  # descending area
    kept: List[Tuple[List[int], int, str]] = []
    for cand in candidates:
        bbox, area, color = cand
        if not any(_iou(bbox, k[0]) > iou_thresh for k in kept):
            kept.append(cand)
    return kept


def _padded_roi(
    bbox: Sequence[int],
    frame_w: int,
    frame_h: int,
    pad: float,
) -> List[int]:
    """
    Expand `bbox` by `pad` fraction on every side, clamped to frame bounds.

    A padded ROI is used for fall detection so that a pin that topples
    slightly sideways (still partially inside the original box) is not
    immediately counted as fallen.
    """
    x, y, w, h = bbox
    px = int(w * pad)
    py = int(h * pad)
    rx = max(0, x - px)
    ry = max(0, y - py)
    rw = min(frame_w - rx, w + 2 * px)
    rh = min(frame_h - ry, h + 2 * py)
    return [rx, ry, rw, rh]


def _count_color_pixels(
    hsv: np.ndarray,
    color_name: str,
    roi: Sequence[int],
) -> int:
    """
    Count pixels of `color_name` inside `roi` = [x, y, w, h].

    Operates on a sub-array of the frame HSV image — fast and allocation-light.
    """
    rx, ry, rw, rh = roi
    patch = hsv[ry : ry + rh, rx : rx + rw]
    if patch.size == 0:
        return 0
    # Build the colour mask for just this patch
    ranges = _HSV_RANGES[color_name]
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv.inRange(patch, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    return int(np.count_nonzero(mask))


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: Tuple[int, int, int],
    font: int = cv.FONT_HERSHEY_SIMPLEX,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw a filled-background text label in-place (top-left anchor at x, y)."""
    (tw, th), baseline = cv.getTextSize(text, font, scale, thickness)
    tx = max(0, x)
    ty = max(th + 6, y)
    # Filled rectangle background
    cv.rectangle(img, (tx, ty - th - 4), (tx + tw + 6, ty + baseline + 2), bg_color, -1)
    # Dark text for legibility on any background colour
    text_color = (0, 0, 0) if sum(bg_color) > 380 else (255, 255, 255)
    cv.putText(img, text, (tx + 3, ty - 2), font, scale, text_color, thickness, cv.LINE_AA)


def _draw_badge(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    text_color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int],
    scale: float = 1.0,
    thickness: int = 2,
    font: int = cv.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Draw a centre-anchored filled badge (for the fall-order number)."""
    (tw, th), baseline = cv.getTextSize(text, font, scale, thickness)
    pad = 6
    x1 = cx - tw // 2 - pad
    y1 = cy - th // 2 - pad
    x2 = cx + tw // 2 + pad
    y2 = cy + th // 2 + pad + baseline
    cv.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv.putText(
        img, text,
        (cx - tw // 2, cy + th // 2),
        font, scale, text_color, thickness, cv.LINE_AA,
    )
