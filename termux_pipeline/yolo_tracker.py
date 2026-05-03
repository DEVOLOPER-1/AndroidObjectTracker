from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnxruntime as ort


@dataclass(slots=True)
class Detection:
	box: tuple[float, float, float, float]
	class_id: int
	confidence: float
	object_type: str


class YOLOTracker:
	def __init__(
		self,
		model_path: str | Path | None = None,
		*,
		input_size: int = 640,
		confidence_threshold: float = 0.4,
		providers: list[str] | None = None,
		save_annotated_image: bool = False,
		annotated_output_dir: str | Path | None = None,
	) -> None:
		self.model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().with_name("yolo26n.onnx")
		self.external_data_path = self.model_path.with_suffix(self.model_path.suffix + ".data")
		self.input_size = input_size
		self.confidence_threshold = confidence_threshold
		self.providers = providers or ["CPUExecutionProvider"]
		self.save_annotated_image = save_annotated_image
		self.annotated_output_dir = Path(annotated_output_dir) if annotated_output_dir is not None else Path(__file__).resolve().parent

		self._validate_model_files()
		self.session = ort.InferenceSession(str(self.model_path), providers=self.providers)
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self._select_detection_output_name()
		self.object_classes = self._build_class_map()

	@staticmethod
	def _build_class_map() -> list[str]:
		return [
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
			"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
			"scissors", "teddy bear", "hair drier", "toothbrush",
		]

	def _validate_model_files(self) -> None:
		if not self.model_path.exists():
			raise FileNotFoundError(f"Missing model file: {self.model_path}")
		if not self.external_data_path.exists():
			raise FileNotFoundError(f"Missing external data file required by the model: {self.external_data_path}")

	def _pil_modules(self) -> tuple[Any, Any, Any]:
		try:
			image_module = import_module("PIL.Image")
			draw_module = import_module("PIL.ImageDraw")
			font_module = import_module("PIL.ImageFont")
			return image_module, draw_module, font_module
		except ModuleNotFoundError as exc:
			raise ImportError("Pillow is required for image loading and annotation saving.") from exc

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
	) -> tuple[np.ndarray, tuple[int, int], Any]:
		image_module, _, _ = self._pil_modules()
		try:
			resample_bilinear = image_module.Resampling.BILINEAR
		except AttributeError:
			resample_bilinear = image_module.BILINEAR

		if isinstance(source, (str, Path)):
			base_image = image_module.open(source).convert("RGB")
			orig_width, orig_height = base_image.size
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
				array = array[:, :, ::-1]

			array = np.clip(array, 0, 255).astype(np.uint8)
			base_image = image_module.fromarray(array, mode="RGB")
			orig_height, orig_width = array.shape[:2]

		resized = base_image.resize((self.input_size, self.input_size), resample_bilinear)
		array = np.asarray(resized, dtype=np.float32) / 255.0
		tensor = np.transpose(array, (2, 0, 1))[None, ...]
		return tensor, (orig_width, orig_height), base_image

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

	def _save_annotated_image(self, image: Any, detections: list[Detection], source_name: str) -> Path:
		_, draw_module, font_module = self._pil_modules()
		annotated = image.copy()
		draw = draw_module.Draw(annotated)
		font = font_module.load_default()

		for detection in detections:
			x1, y1, x2, y2 = detection.box
			label = f"{detection.object_type} {detection.confidence:.2f}"
			draw.rectangle((x1, y1, x2, y2), outline="red", width=3)

			try:
				left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
				text_width = right - left
				text_height = bottom - top
			except Exception:
				text_width, text_height = font.getsize(label)

			text_x = max(0, int(x1))
			text_y = max(0, int(y1) - text_height - 4)
			bg_box = (text_x, text_y, text_x + text_width + 6, text_y + text_height + 4)
			draw.rectangle(bg_box, fill="red")
			draw.text((text_x + 3, text_y + 2), label, fill="white", font=font)

		output_path = self.annotated_output_dir / f"annotated_{source_name}.jpg"
		annotated.save(output_path)
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