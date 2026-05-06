"""
pipeline.py — State logic: pin-fall detection, trajectory, timers, run summary.

Coordinate convention
---------------------
All bounding boxes stored in ``PinState`` and passed to helpers in this module
are in **(x, y, w, h)** format.  YOLO detections arrive in **(x1, y1, x2, y2)**
format; the conversion is performed once in ``_xyxy_to_xywh`` the moment a
``Detection`` bbox is committed to a ``PinState``.

Android compatibility
---------------------
``PinFallTracker.get_state_dict()`` returns a plain dict/list tree with no
NumPy or OpenCV types — safe to serialise with ``json.dumps`` and pass across
JNI / BroadcastReceiver boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from yolo_tracker import Detection, TrackedDetection


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PinState:
    track_id: int
    bbox: tuple[int, int, int, int]           # (x, y, w, h)  ← xywh always
    color_name: str = "pin"
    draw_bgr: tuple[int, int, int] = (255, 255, 255)
    roi: tuple[int, int, int, int] | None = None
    reference_bbox: tuple[int, int, int, int] | None = None
    is_fallen: bool = False
    fall_order: int = 0                       # 1-based; 0 = still standing
    last_seen_frame: int = 0
    palette_index: int = 0


@dataclass(slots=True)
class RunSummary:
    elapsed_seconds: float
    pins_fallen: int
    total_pins: int


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xyxy_to_xywh(
    box: tuple[float, float, float, float] | Sequence[float],
) -> tuple[int, int, int, int]:
    """Convert a YOLO xyxy box to (x, y, w, h) integers.

    This is the single conversion point.  Call it once when a detection bbox
    enters PinState; never store raw xyxy in PinState.
    """
    x1, y1, x2, y2 = box
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2 - x1)),
        int(round(y2 - y1)),
    )


def center_from_bbox(bbox: Sequence[int] | Sequence[float]) -> tuple[int, int]:
    """Return (cx, cy) centre pixel from an xywh bbox."""
    if len(bbox) != 4:
        raise ValueError("Expected bbox with four values")
    x, y, w, h = [int(round(v)) for v in bbox]
    return x + w // 2, y + h // 2


def pick_primary_detection(
    detections: Sequence[Detection | TrackedDetection],
    preferred_class: str,
) -> Detection | TrackedDetection | None:
    if not detections:
        return None
    preferred = [d for d in detections if getattr(d, "object_type", None) == preferred_class]
    if not preferred:
        return None
    return max(preferred, key=lambda d: float(getattr(d, "confidence", 0.0)))


def _bbox_iou(
    box_a: Sequence[int] | Sequence[float],
    box_b: Sequence[int] | Sequence[float],
) -> float:
    """IoU between two *xywh* bounding boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0.0 else 0.0


def make_pin_roi(
    bbox: Sequence[int] | Sequence[float],
    *,
    expand_ratio: float = 0.18,
) -> tuple[int, int, int, int]:
    """Return a padded xywh ROI around *bbox* (also xywh)."""
    if len(bbox) != 4:
        raise ValueError("Expected bbox with four values")
    x, y, w, h = [int(round(v)) for v in bbox]
    pad_x = max(2, int(round(w * expand_ratio)))
    pad_y = max(2, int(round(h * expand_ratio)))
    return max(0, x - pad_x), max(0, y - pad_y), w + pad_x * 2, h + pad_y * 2


# ─────────────────────────────────────────────────────────────────────────────
# Fall-detection heuristic
# ─────────────────────────────────────────────────────────────────────────────

def _intersection_area(
    box_a: Sequence[int] | Sequence[float],
    box_b: Sequence[int] | Sequence[float],
) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    ax, ay, aw, ah = [float(v) for v in box_a]
    bx, by, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def infer_pin_is_fallen(
    pin_bbox: Sequence[int] | Sequence[float],
    car_bbox: Sequence[int] | Sequence[float] | None,
) -> bool:
    """Return True when the car bbox overlaps the pin AND the pin has fallen sideways.

    The “fallen” condition is: overlap exists **and** the pin’s detected
    bounding box is wider than it is tall (i.e. it has tipped over).
    """
    if car_bbox is None or len(pin_bbox) != 4 or len(car_bbox) != 4:
        return False

    # 1. Must overlap the car
    if _intersection_area(pin_bbox, car_bbox) <= 0.0:
        return False

    # 2. Must have a sideways shape (fallen)
    _, _, w, h = pin_bbox
    if w <= h:                 # still upright
        return False

    return True

# ─────────────────────────────────────────────────────────────────────────────
# PinFallTracker
# ─────────────────────────────────────────────────────────────────────────────

class PinFallTracker:
    """
    Stateful tracker for bowling-pin fall events.

    All ``Detection.box`` values are expected in **xyxy** format (as produced by
    ``YOLOTracker``).  The tracker converts them to **xywh** internally before
    storing in ``PinState.bbox``.
    """

    def __init__(
        self,
        max_missing_frames: int = 10,
    ) -> None:
        self.max_missing_frames  = max_missing_frames

        self._pins_by_track_id: dict[int, PinState] = {}
        self._palette_counter: int = 0
        self._next_fall_order: int = 1

    def initialize(
        self,
        pin_detections: Sequence[Detection | TrackedDetection],
        *,
        car_bbox: Sequence[int] | Sequence[float] | None = None,
        frame_index: int = 0,
    ) -> list[PinState]:
        """Reset and register frame-0 detections as standing pins."""
        self._pins_by_track_id.clear()
        self._palette_counter = 0
        self._next_fall_order = 1
        return self.update(pin_detections, car_bbox=car_bbox, frame_index=frame_index)

    def update(
        self,
        pin_detections: Sequence[Detection | TrackedDetection],
        *,
        car_bbox: Sequence[int] | Sequence[float] | None = None,
        frame_index: int,
    ) -> list[PinState]:
        """
        Ingest current-frame detections, update state, return all known pins.

        Each pin becomes fallen the first time its bbox overlaps the car bbox.
        Standing pins remain unlabeled in the renderer; only ``fall_order`` is
        displayed once a pin has fallen.
        """
        for index, detection in enumerate(pin_detections):
            track_id = int(getattr(detection, "track_id", index + 1))

            # ── THE critical fix: xyxy → xywh ──────────────────────────────
            bbox_xywh = _xyxy_to_xywh(detection.box)

            state = self._pins_by_track_id.get(track_id)

            if state is None:
                state = PinState(
                    track_id=track_id,
                    bbox=bbox_xywh,
                    color_name="pin",
                    draw_bgr=(255, 255, 255),
                    reference_bbox=bbox_xywh,
                    roi=make_pin_roi(bbox_xywh),
                    last_seen_frame=frame_index,
                )
                self._pins_by_track_id[track_id] = state
            else:
                if not state.is_fallen:
                    state.bbox = bbox_xywh
                    state.roi  = make_pin_roi(bbox_xywh)
                state.last_seen_frame = frame_index

            if not state.is_fallen:
                if infer_pin_is_fallen(bbox_xywh, car_bbox):
                    state.is_fallen  = True
                    state.fall_order = self._next_fall_order
                    self._next_fall_order += 1

        return self.get_pins()

    def get_pins(self) -> list[PinState]:
        return [self._pins_by_track_id[k] for k in sorted(self._pins_by_track_id)]

    def fallen_count(self) -> int:
        return sum(1 for p in self._pins_by_track_id.values() if p.is_fallen)

    def total_pins(self) -> int:
        return len(self._pins_by_track_id)

    def summary(self, elapsed_seconds: float) -> RunSummary:
        return RunSummary(
            elapsed_seconds=elapsed_seconds,
            pins_fallen=self.fallen_count(),
            total_pins=self.total_pins(),
        )

    def get_state_dict(self, elapsed_seconds: float = 0.0) -> dict:
        """
        JSON-serialisable snapshot — no NumPy, no OpenCV.
        Used by the Android bridge.
        """
        return {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "pins_fallen":     self.fallen_count(),
            "total_pins":      self.total_pins(),
            "pins": [
                {
                    "track_id":        p.track_id,
                    "bbox":            list(p.bbox),      # [x, y, w, h]
                    "color_name":      p.color_name,
                    "draw_bgr":        list(p.draw_bgr),  # [b, g, r]
                    "is_fallen":       p.is_fallen,
                    "fall_order":      p.fall_order,      # 0 = still standing
                    "last_seen_frame": p.last_seen_frame,
                }
                for p in self.get_pins()
            ],
        }



# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    from pipeline_orchestrator import PipelineConfig, run_pipeline

    parser = argparse.ArgumentParser(
        description="Run YOLO detection + pin-fall tracking over a video."
    )
    parser.add_argument("video_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--no-display",      action="store_true")
    parser.add_argument("--process-every",   type=int,   default=1)
    parser.add_argument("--trail-length",    type=int,   default=200)
    parser.add_argument("--trail-min-alpha", type=float, default=0.25)
    parser.add_argument("--preferred-class", type=str,   default="car")
    parser.add_argument("--max-search-frames", type=int, default=600)
    parser.add_argument("--show-pin-roi",    action="store_true")
    parser.add_argument("--no-score",        action="store_true")
    parser.add_argument(
        "--nms-score-threshold", type=float, default=0.25,
        help="YOLO confidence threshold (0.0-1.0). Higher filters low-confidence boxes. "
             "Try 0.35-0.50 for stricter, 0.10-0.20 for lenient. Default: 0.25"
    )
    parser.add_argument(
        "--nms-iou-threshold", type=float, default=0.45,
        help="NMS IoU overlap threshold (0.0-1.0). Lower=more aggressive dedup (0.25-0.35), "
             "Higher=more lenient (0.55-0.65). Default: 0.45"
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
        show_pin_roi=args.show_pin_roi,
        show_score=not args.no_score,
        nms_score_threshold=args.nms_score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )
    output_path = run_pipeline(config)
    print(f"Saved tracked video to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python pipeline.py video.mp4 --no-display --process-every 3 --trail-length 300 --trail-min-alpha 0.2

# python pipeline.py IMG_3778.MP4 --max-search-frames 600  # Search 20 seconds instead