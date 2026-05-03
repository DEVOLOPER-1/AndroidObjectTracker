"""
AbaVi tracker wrapper.

- Exposes get_tracker_class() so your evaluation harness can import it.
- Tracker class signature: Tracker(params, dataset_name)
- Methods used by harness:
    - initialize(frame, init_info) -> dict (may include 'target_bbox')
    - track(frame, info) -> dict with 'target_bbox' and optionally timing

Behavior:
- Attempts to load an ONNX model (AbaViTrack.onnx) colocated with this module.
- If ONNX I/O is unknown/unusable, falls back to OpenCV template matching.
- Accepts frames as numpy arrays in RGB (this matches your abavit_tracker._read_image which returns RGB).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import cv2 as cv

try:
    import onnxruntime as ort
except Exception:
    ort = None  # ONNX not available; fallback to CV template-matching


class _ONNXWrapper:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.external_data_path = model_path.with_suffix(model_path.suffix + ".data")
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if self.external_data_path.exists() is False and any(s.endswith(".data") for s in [str(self.model_path.with_suffix(".onnx.data"))]):
            # if no external data, proceed — depends on model
            pass

        if ort is None:
            raise RuntimeError("onnxruntime not available")

        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.inputs = {inp.name: inp for inp in self.session.get_inputs()}
        self.outputs = {out.name: out for out in self.session.get_outputs()}
        self.template_input_name = "template" if "template" in self.inputs else self.session.get_inputs()[0].name
        self.search_input_name = "search" if "search" in self.inputs else self.session.get_inputs()[-1].name
        self.pred_boxes_output_name = "pred_boxes" if "pred_boxes" in self.outputs else self.session.get_outputs()[0].name

    @staticmethod
    def _prepare_tensor(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        width, height = size
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        resized = cv.resize(image_rgb, (width, height), interpolation=cv.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def predict(self, template_rgb: np.ndarray, search_rgb: np.ndarray) -> np.ndarray:
        template = self._prepare_tensor(template_rgb, (128, 128))
        search = self._prepare_tensor(search_rgb, (256, 256))
        out = self.session.run(
            [self.pred_boxes_output_name],
            {self.template_input_name: template, self.search_input_name: search},
        )
        return np.asarray(out[0])


class AbaViTracker:
    """
    Tracker wrapper used by your evaluation harness.

    Usage:
      tracker = AbaViTracker(params, dataset_name)
      tracker.initialize(frame, {'init_bbox': [x,y,w,h]})
      out = tracker.track(frame, info)
    """

    multiobj_mode = "default"  # used by caller

    def __init__(self, params: Any, dataset_name: str):
        self.params = params
        self.dataset_name = dataset_name
        self.prev_bbox = None  # (x,y,w,h)
        self.last_time = None
        self.template = None
        self.model_wrapper = None
        self.template_tensor = None
        # default model path next to this file
        self.model_path = Path(__file__).resolve().with_name("AbaViTrack.onnx")
        # try to load ONNX; if fails, we'll use template-matching fallback
        try:
            if ort is not None and self.model_path.exists():
                self.model_wrapper = _ONNXWrapper(self.model_path)
        except Exception as exc:
            # ONNX not usable — fall back to CV
            print(f"[AbaViTracker] ONNX load failed, fallback to template-matching: {exc}")
            self.model_wrapper = None

    def initialize(self, frame: np.ndarray, init_info: dict):
        """
        Initialize tracker with the first frame and init_info dict containing 'init_bbox': [x,y,w,h].
        Return an initial output dict (may include 'target_bbox').
        """
        bbox = init_info.get("init_bbox")
        if bbox is None:
            raise ValueError("init_info must include 'init_bbox'")

        bbox_values = list(cast(Sequence[float], bbox))
        x, y, w, h = [int(v) for v in bbox_values]
        self.prev_bbox = (x, y, w, h)
        # store template for CV fallback (RGB)
        self.template = self._extract_patch(frame, self.prev_bbox)
        self.template_tensor = self._build_template_tensor(self.template)
        self.last_time = time.time()

        # If ONNX available, we might optionally warm it with the initial patch (not required).
        return {"target_bbox": [x, y, w, h], "time": 0.0}

    def _build_template_tensor(self, template_patch: np.ndarray) -> np.ndarray:
        if template_patch is None or template_patch.size == 0:
            return np.zeros((1, 3, 128, 128), dtype=np.float32)
        if template_patch.dtype != np.uint8:
            template_patch = np.clip(template_patch, 0, 255).astype(np.uint8)
        resized = cv.resize(template_patch, (128, 128), interpolation=cv.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32)

    def _extract_patch(self, frame: np.ndarray, bbox: tuple[int, int, int, int], pad: float = 0.0) -> np.ndarray:
        x, y, w, h = bbox
        if pad > 0:
            xp = max(0, int(x - w * pad))
            yp = max(0, int(y - h * pad))
            wp = min(frame.shape[1] - xp, int(w * (1 + 2 * pad)))
            hp = min(frame.shape[0] - yp, int(h * (1 + 2 * pad)))
            x, y, w, h = xp, yp, wp, hp
        # ensure inside frame
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        patch = frame[y1:y2, x1:x2].copy()
        return patch

    def _template_track(self, frame: np.ndarray, search_scale: float = 2.0) -> tuple[int, int, int, int]:
        """
        Template-match in a search window around previous bbox.
        Returns new bbox (x,y,w,h)
        """
        x, y, w, h = self.prev_bbox
        cx = x + w // 2
        cy = y + h // 2
        search_w = int(w * search_scale)
        search_h = int(h * search_scale)
        sx1 = max(0, int(cx - search_w // 2))
        sy1 = max(0, int(cy - search_h // 2))
        sx2 = min(frame.shape[1], sx1 + search_w)
        sy2 = min(frame.shape[0], sy1 + search_h)
        search_region = frame[sy1:sy2, sx1:sx2]
        if search_region.size == 0 or self.template is None or self.template.size == 0:
            return self.prev_bbox

        # convert both to grayscale for matching
        templ_gray = cv.cvtColor(self.template, cv.COLOR_RGB2GRAY) if self.template.ndim == 3 else self.template
        search_gray = cv.cvtColor(search_region, cv.COLOR_RGB2GRAY) if search_region.ndim == 3 else search_region

        # ensure template smaller than search region; if not, fall back to full-frame search
        if templ_gray.shape[0] > search_gray.shape[0] or templ_gray.shape[1] > search_gray.shape[1]:
            search_gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            sx1, sy1 = 0, 0

        res = cv.matchTemplate(search_gray, templ_gray, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = (sx1 + max_loc[0], sy1 + max_loc[1])
        new_x, new_y = top_left
        new_w, new_h = self.template.shape[1], self.template.shape[0]
        return int(new_x), int(new_y), int(new_w), int(new_h)

    def _onnx_track(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """
        Use the ONNX model to predict new bbox relative to the whole frame.
        This is model-dependent: we try to give a reasonable default behavior:
        - create a patch around the previous bbox (with some padding), resize to model input,
          run model and decode the output to a bbox.
        - if decoding cannot be confidently done, return None (so fallback used).
        """
        if self.model_wrapper is None:
            return None

        if self.template is None or self.template.size == 0:
            return None

        try:
            out = self.model_wrapper.predict(self.template, frame)  # raw prediction
        except Exception as exc:
            print(f"[AbaViTracker] ONNX inference failed: {exc}")
            return None

        arr = np.asarray(out)
        if arr.size < 4:
            return None

        box = np.asarray(arr.reshape(-1, 4)[0], dtype=np.float32)
        return self._decode_pred_box(box, frame.shape[1], frame.shape[0])

    def _decode_pred_box(self, box: np.ndarray, frame_w: int, frame_h: int) -> tuple[int, int, int, int] | None:
        if box.size < 4:
            return None

        x1, y1, x2, y2 = map(float, box[:4])
        if max(x1, y1, x2, y2) <= 1.5:
            # normalized center format from this model: [cx, cy, w, h]
            cx, cy, bw, bh = x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
        elif x2 <= x1 or y2 <= y1:
            # center format in absolute pixels
            cx, cy, bw, bh = x1, y1, x2, y2
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0

        x1 = float(np.clip(x1, 0.0, frame_w - 1.0))
        y1 = float(np.clip(y1, 0.0, frame_h - 1.0))
        x2 = float(np.clip(x2, x1 + 1.0, frame_w))
        y2 = float(np.clip(y2, y1 + 1.0, frame_h))
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    def track(self, frame: np.ndarray, info: dict | None = None) -> dict:
        """
        Track one frame. Returns a dict with at least 'target_bbox' (list [x,y,w,h]).
        'info' can contain previous_output or other info from harness.
        """
        t0 = time.time()
        if self.prev_bbox is None:
            # not initialized
            return {"target_bbox": None, "time": 0.0}

        # First try ONNX model if available
        new_bbox = None
        if self.model_wrapper is not None:
            try:
                new_bbox = self._onnx_track(frame)
            except Exception as exc:
                print(f"[AbaViTracker] ONNX tracking error: {exc}")
                new_bbox = None

        if new_bbox is None:
            # fallback to template matching
            new_bbox = self._template_track(frame, search_scale=2.0)

        # update internal state
        self.prev_bbox = new_bbox
        # update template adaptively: small update to template (optional)
        try:
            # refresh template slowly (simple linear update)
            new_template = self._extract_patch(frame, new_bbox)
            if new_template.size > 0 and self.template is not None and self.template.size > 0:
                alpha = 0.1
                # resize new_template to template size if different
                if new_template.shape != self.template.shape:
                    new_template = cv.resize(new_template, (self.template.shape[1], self.template.shape[0]), interpolation=cv.INTER_LINEAR)
                self.template = cv.addWeighted(self.template.astype(np.float32), 1.0 - alpha, new_template.astype(np.float32), alpha, 0.0).astype(np.uint8)
            elif new_template.size > 0:
                self.template = new_template
        except Exception:
            pass

        t1 = time.time()
        return {"target_bbox": [int(new_bbox[0]), int(new_bbox[1]), int(new_bbox[2]), int(new_bbox[3])], "time": t1 - t0}


# helper required by your harness: return the tracker class
def get_tracker_class():
    return AbaViTracker

