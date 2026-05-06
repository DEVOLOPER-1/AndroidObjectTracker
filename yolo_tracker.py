from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import cv2 as cv
import numpy as np
import onnxruntime as ort


# -------------------------
# Tweakable defaults for YOLO
# -------------------------
MODEL_FILENAME: str = "yolo26n.onnx"
EXTERNAL_DATA_SUFFIX: str = ".data"
INPUT_SIZE_DEFAULT: int = 640
CONFIDENCE_THRESHOLD_DEFAULT: float = 0.1
PROVIDERS_DEFAULT: list[str] = [
	"CUDAExecutionProvider",
	"CoreMLExecutionProvider",
	"CPUExecutionProvider",
]


@dataclass(slots=True)
class Detection:
	box: tuple[float, float, float, float]
	class_id: int
	confidence: float
	object_type: str


@dataclass(slots=True)
class TrackedDetection(Detection):
	track_id: int = 0


@dataclass(slots=True)
class _TrackState:
	track_id: int
	box: tuple[float, float, float, float]
	class_id: int
	confidence: float
	object_type: str
	missed_frames: int = 0
	history: list[tuple[int, int]] = field(default_factory=list)


class YOLOTracker:
	def __init__(
		self,
		model_path: str | Path | None = None,
		*,
		input_size: int = INPUT_SIZE_DEFAULT,
		confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
		providers: list[str] | None = None,
		save_annotated_image: bool = False,
		annotated_output_dir: str | Path | None = None,
	) -> None:
		self.model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().with_name(MODEL_FILENAME)
		self.external_data_path = self.model_path.with_suffix(self.model_path.suffix + EXTERNAL_DATA_SUFFIX)
		self.input_size = input_size
		self.confidence_threshold = confidence_threshold
		self.providers = providers or PROVIDERS_DEFAULT
		self.save_annotated_image = save_annotated_image
		self.annotated_output_dir = Path(annotated_output_dir) if annotated_output_dir is not None else Path(__file__).resolve().parent
		self.match_iou_threshold = 0.50
		self.max_track_age = 15
		self._frame_index = 0
		self._next_track_id = 1
		self._tracks: dict[int, _TrackState] = {}
		self.nms_score_threshold: float = 0.25
		self.nms_iou_threshold: float = 0.45

		self._validate_model_files()
		self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self._select_detection_output_name()
		self.object_classes = self._build_class_map()
		print(f"[YOLOTracker] Loaded {self.model_path.name} with providers: {self.providers}")

	@staticmethod
	def _build_class_map() -> list[str]:
		return ["bowling-ball", "bowling-pins", "sweep board", "car"] #**Classes:** `0: bowling-ball`, `1: bowling-pins`, `2: sweep board`, `3: car`

	def _validate_model_files(self) -> None:
		if not self.model_path.exists():
			raise FileNotFoundError(f"Missing model file: {self.model_path}")
		if not self.external_data_path.exists():
			raise FileNotFoundError(f"Missing external data file required by the model: {self.external_data_path}")

	def _select_detection_output_name(self) -> str:
		outputs = self.session.get_outputs()
		for output in outputs:
			if output.name.lower() == "output":
				return output.name

		candidates: list[str] = []
		for output in outputs:
			shape = output.shape
			if len(shape) not in (2, 3):
				continue
			last_dim = shape[-1]
			if isinstance(last_dim, int) and last_dim >= 6:
				candidates.append(output.name)

		return candidates[0] if candidates else outputs[0].name

	def _load_image(
		self,
		source: np.ndarray | str | Path,
		*,
		is_bgr: bool,
	) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:

		if isinstance(source, (str, Path)):
			base_bgr = cv.imread(str(source), cv.IMREAD_COLOR)
			if base_bgr is None:
				raise FileNotFoundError(f"Unable to read image: {source}")
		else:
			array = np.asarray(source)
			if array.ndim == 2:
				array = np.repeat(array[:, :, None], 3, axis=2)
			elif array.ndim != 3:
				raise ValueError("Input image/frame must have 2 or 3 dimensions")

			if array.shape[2] == 4:
				array = array[:, :, :3]
			elif array.shape[2] != 3:
				raise ValueError("Input image/frame must have 1, 3, or 4 channels")

			if is_bgr:
				base_bgr = np.clip(array, 0, 255).astype(np.uint8)
			else:
				base_bgr = np.clip(array[:, :, ::-1], 0, 255).astype(np.uint8)

		orig_height, orig_width = base_bgr.shape[:2]
		resized_bgr = cv.resize(base_bgr, (self.input_size, self.input_size), interpolation=cv.INTER_LINEAR)
		resized_rgb = cv.cvtColor(resized_bgr, cv.COLOR_BGR2RGB)
		array = resized_rgb.astype(np.float32) / 255.0
		tensor = np.transpose(array, (2, 0, 1))[None, ...]
		return tensor, (orig_width, orig_height), base_bgr

	def _to_xyxy(self, box: np.ndarray) -> np.ndarray:
		x1, y1, x2, y2 = map(float, box[:4])
		if x2 < x1 or y2 < y1:
			cx, cy, w, h = map(float, box[:4])
			x1 = cx - w / 2.0
			y1 = cy - h / 2.0
			x2 = cx + w / 2.0
			y2 = cy + h / 2.0
		return np.array([x1, y1, x2, y2], dtype=np.float32)

	def _scale_box(self, box: np.ndarray, original_size: tuple[int, int]) -> tuple[float, float, float, float]:
		orig_width, orig_height = original_size
		if np.max(box) <= 1.5:
			box = box.copy()
			box[0::2] *= self.input_size
			box[1::2] *= self.input_size

		scale_x = orig_width / float(self.input_size)
		scale_y = orig_height / float(self.input_size)
		box = box.copy()
		box[0] *= scale_x
		box[2] *= scale_x
		box[1] *= scale_y
		box[3] *= scale_y
		box[0] = float(np.clip(box[0], 0.0, orig_width))
		box[2] = float(np.clip(box[2], 0.0, orig_width))
		box[1] = float(np.clip(box[1], 0.0, orig_height))
		box[3] = float(np.clip(box[3], 0.0, orig_height))
		return float(box[0]), float(box[1]), float(box[2]), float(box[3])

	@staticmethod
	def _bbox_center(box: tuple[float, float, float, float]) -> tuple[int, int]:
		x1, y1, x2, y2 = box
		return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))

	@staticmethod
	def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
		ax1, ay1, ax2, ay2 = box_a
		bx1, by1, bx2, by2 = box_b

		inter_x1 = max(ax1, bx1)
		inter_y1 = max(ay1, by1)
		inter_x2 = min(ax2, bx2)
		inter_y2 = min(ay2, by2)
		if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
			return 0.0

		inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
		area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
		area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
		union = area_a + area_b - inter_area
		if union <= 0.0:
			return 0.0
		return float(inter_area / union)

	@staticmethod
	def _apply_nms(
		detections: list[Detection],
		score_threshold: float = 0.25,
		iou_threshold: float = 0.45,
	) -> list[Detection]:
		"""Apply OpenCV-native NMS for efficient deduplication."""
		if len(detections) <= 1:
			return detections

		boxes = np.array([list(d.box) for d in detections], dtype=np.float32)
		scores = np.array([d.confidence for d in detections], dtype=np.float32)
		indices = cv.dnn.NMSBoxes(
			bboxes=boxes,
			scores=scores,
			score_threshold=score_threshold,
			nms_threshold=iou_threshold,
		)
		return [detections[i] for i in indices]

	def _match_tracks(self, detections: list[Detection]) -> list[TrackedDetection]:
		remaining_tracks = dict(self._tracks)
		assigned_track_ids: set[int] = set()
		tracked_detections: list[TrackedDetection] = []

		for detection in sorted(detections, key=lambda item: float(item.confidence), reverse=True):
			best_track_id = None
			best_iou = 0.0
			for track_id, track_state in remaining_tracks.items():
				if track_id in assigned_track_ids:
					continue
				if track_state.class_id != detection.class_id or track_state.object_type != detection.object_type:
					continue
				score = self._iou(track_state.box, detection.box)
				if score > best_iou:
					best_iou = score
					best_track_id = track_id

			if best_track_id is not None and best_iou >= self.match_iou_threshold:
				track_state = self._tracks[best_track_id]
				track_state.box = detection.box
				track_state.confidence = detection.confidence
				track_state.missed_frames = 0
				track_state.history.append(self._bbox_center(detection.box))
				assigned_track_ids.add(best_track_id)
				tracked_detections.append(
					TrackedDetection(
						box=detection.box,
						class_id=detection.class_id,
						confidence=detection.confidence,
						object_type=detection.object_type,
						track_id=best_track_id,
					)
				)
			else:
				track_id = self._next_track_id
				self._next_track_id += 1
				self._tracks[track_id] = _TrackState(
					track_id=track_id,
					box=detection.box,
					class_id=detection.class_id,
					confidence=detection.confidence,
					object_type=detection.object_type,
					history=[self._bbox_center(detection.box)],
				)
				assigned_track_ids.add(track_id)
				tracked_detections.append(
					TrackedDetection(
						box=detection.box,
						class_id=detection.class_id,
						confidence=detection.confidence,
						object_type=detection.object_type,
						track_id=track_id,
					)
				)

		for track_id, track_state in list(self._tracks.items()):
			if track_id not in assigned_track_ids:
				track_state.missed_frames += 1
				if track_state.missed_frames > self.max_track_age:
					del self._tracks[track_id]

		tracked_detections.sort(key=lambda item: item.track_id)
		return tracked_detections

	def track(
			self,
			source: np.ndarray | str | Path,
			*,
			is_bgr: bool = True,
			save_annotated_image: bool | None = None,
	) -> list[TrackedDetection]:
		detections = self.infer(source, is_bgr=is_bgr, save_annotated_image=save_annotated_image)

		self._frame_index += 1
		return self._match_tracks(detections)

	def get_track_history(self, track_id: int) -> list[tuple[int, int]]:
		track = self._tracks.get(track_id)
		return list(track.history) if track is not None else []

	def get_active_tracks(self) -> list[TrackedDetection]:
		return [
			TrackedDetection(
				box=track.box,
				class_id=track.class_id,
				confidence=track.confidence,
				object_type=track.object_type,
				track_id=track.track_id,
			)
			for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
		]

	def _decode(self, predictions: np.ndarray, original_size: tuple[int, int]) -> list[Detection]:
		predictions = np.asarray(predictions)
		if predictions.ndim == 3:
			predictions = predictions[0]
		elif predictions.ndim == 1:
			predictions = predictions[None, :]

		detections: list[Detection] = []
		for row in predictions:
			if row.shape[0] < 6:
				continue

			box = self._to_xyxy(row[:4])
			confidence = float(row[4])
			class_id = int(round(float(row[5])))

			if row.shape[0] > 6:
				class_scores = row[5:]
				class_id = int(np.argmax(class_scores))
				confidence = float(row[4] * class_scores[class_id])

			if confidence < self.confidence_threshold:
				continue

			scaled_box = self._scale_box(box, original_size)
			object_type = self.object_classes[class_id] if 0 <= class_id < len(self.object_classes) else f"class_{class_id}"
			detections.append(
				Detection(
					box=scaled_box,
					class_id=class_id,
					confidence=confidence,
					object_type=object_type,
				)
			)

		return detections

	def _save_annotated_image(self, image: np.ndarray, detections: list[Detection], source_name: str) -> Path:
		annotated = image.copy()
		font = cv.FONT_HERSHEY_SIMPLEX

		for detection in detections:
			x1, y1, x2, y2 = [int(round(v)) for v in detection.box]
			label = f"{detection.object_type} {detection.confidence:.2f}"
			cv.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
			(text_w, text_h), baseline = cv.getTextSize(label, font, 0.6, 2)
			text_x = max(0, x1)
			text_y = max(text_h + 6, y1 - 6)
			cv.rectangle(
				annotated,
				(text_x, text_y - text_h - baseline - 4),
				(text_x + text_w + 6, text_y + baseline),
				(0, 0, 255),
				thickness=-1,
			)
			cv.putText(annotated, label, (text_x + 3, text_y - 4), font, 0.6, (255, 255, 255), 2, cv.LINE_AA)

		output_path = self.annotated_output_dir / f"annotated_{source_name}.jpg"
		cv.imwrite(str(output_path), annotated)
		print(f"Annotated image saved to: {output_path}")
		return output_path

	def infer(
		self,
		source: np.ndarray | str | Path,
		*,
		is_bgr: bool = True,
		save_annotated_image: bool | None = None,
	) -> list[Detection]:
		tensor, original_size, base_image = self._load_image(source, is_bgr=is_bgr)
		raw_predictions = self.session.run([self.output_name], {self.input_name: tensor})[0]
		predictions = np.asarray(cast(np.ndarray, raw_predictions))
		detections = self._decode(predictions, original_size)

		# ── Apply OpenCV NMS for efficient deduplication ─────────────────────
		detections = self._apply_nms(detections, self.nms_score_threshold, self.nms_iou_threshold)

		print(f"Detected {len(detections)} object(s)")
		for detection in detections:
			x1, y1, x2, y2 = detection.box
			print(
				f"class={detection.class_id} ({detection.object_type}) confidence={detection.confidence:.3f} "
				f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
			)

		should_save = self.save_annotated_image if save_annotated_image is None else save_annotated_image
		if should_save:
			source_name = Path(source).stem if isinstance(source, (str, Path)) else "frame"
			self._save_annotated_image(base_image, detections, source_name)

		return detections

	__call__ = infer


def _main(argv: list[str] | None = None) -> int:
	import argparse

	parser = argparse.ArgumentParser(description="Run YOLO ONNX inference.")
	parser.add_argument("image", help="Path to an image file")
	parser.add_argument("--save-annotated-image", action="store_true", help="Save annotated output beside yolo_tracker.py")
	args = parser.parse_args(argv)

	tracker = YOLOTracker(save_annotated_image=args.save_annotated_image)
	tracker.infer(args.image, is_bgr=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(_main())


# python yolo_tracker.py rc-car.png --save-annotated-image