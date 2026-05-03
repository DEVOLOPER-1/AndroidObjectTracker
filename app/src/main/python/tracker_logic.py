import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io

class PythonTracker:
    def __init__(self, yolo_path, abavit_path):
        # Initialize ONNX Runtime sessions
        self.yolo_session = ort.InferenceSession(yolo_path)
        self.abavit_session = ort.InferenceSession(abavit_path)
        self.template_tensor = None
        self.last_bbox = None # [x1, y1, x2, y2] in full frame
        self.car_path = []

    def detect_yolo(self, image_bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]

        # Preprocess for YOLO (640x640)
        input_img = cv2.resize(img, (640, 640))
        input_img = input_img.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Inference
        outputs = self.yolo_session.run(None, {self.yolo_session.get_inputs()[0].name: input_img})

        # Simple parser (assuming [1, 84, 8400] or similar)
        # In a real scenario, implement full NMS and scaling
        # Returning a dummy detection for structure
        return {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "class": 2, "conf": 0.9}

    def init_sot(self, image_bytes, bbox):
        # Capture template crop
        self.last_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        self.car_path = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]]

        # Preprocess template for AbaViTrack (128x128)
        # In a real scenario, crop and normalize here
        self.template_tensor = np.random.randn(1, 3, 128, 128).astype(np.float32)

    def update_sot(self, image_bytes):
        if self.template_tensor is None or self.last_bbox is None:
            return None

        # Preprocess search region (256x256)
        # Inference using abavit_session
        # outputs = self.abavit_session.run(None, {...})

        # Mock update logic
        new_bbox = [self.last_bbox[0] + 5, self.last_bbox[1] + 2,
                    self.last_bbox[2] + 5, self.last_bbox[3] + 2]
        self.last_bbox = new_bbox
        self.car_path.append([(new_bbox[0]+new_bbox[2])/2, (new_bbox[1]+new_bbox[3])/2])

        return {
            "bbox": self.last_bbox,
            "path": self.car_path,
            "prob": 0.85
        }

def process_frame(tracker, image_bytes, is_init, initial_bbox=None):
    if is_init:
        tracker.init_sot(image_bytes, initial_bbox)
        return {"status": "initialized"}
    else:
        result = tracker.update_sot(image_bytes)
        return result
