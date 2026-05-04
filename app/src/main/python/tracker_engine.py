import numpy as np
import cv2

class TrackerEngine:
    def __init__(self):
        self.template_tensor = None
        self.last_bbox = None # [x1, y1, x2, y2]
        self.car_path = []
        self.conf_threshold = 0.25
        
        # Hanning window initialization
        self.grid_dim = 16
        hann1d = np.hanning(self.grid_dim)
        self.hanning_window = np.outer(hann1d, hann1d).flatten()

    def init_tracker_preprocess(self, image_bytes, bbox):
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None

        self.last_bbox = bbox
        self.car_path = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]]

        # Extract template crop (128x128)
        side = max(bbox[2]-bbox[0], bbox[3]-bbox[1]) * 2.0
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        
        template_crop = self.get_crop(img, cx, cy, side, 128)
        self.template_tensor = self.normalize(template_crop)
        return self.template_tensor

    def update_preprocess(self, image_bytes):
        if self.template_tensor is None or self.last_bbox is None:
            return None, 0, 0, 0

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return None, 0, 0, 0

        # Extract search region (256x256) centered on last bbox
        side = max(self.last_bbox[2]-self.last_bbox[0], self.last_bbox[3]-self.last_bbox[1]) * 4.0
        cx, cy = (self.last_bbox[0]+self.last_bbox[2])/2, (self.last_bbox[1]+self.last_bbox[3])/2
        
        search_crop = self.get_crop(img, cx, cy, side, 256)
        search_tensor = self.normalize(search_crop)
        
        return search_tensor, cx, cy, side

    def update_postprocess(self, outputs_list, cx, cy, side):
        # outputs_list is a list of numpy arrays from ONNX
        score_map = np.array(outputs_list[0]).flatten()
        
        # Apply Hanning window
        alpha = 0.4
        windowed_scores = (1 - alpha) * score_map + alpha * self.hanning_window
        best_idx = np.argmax(windowed_scores)
        
        prob = 1.0 / (1.0 + np.exp(-score_map[best_idx]))
        if prob < self.conf_threshold:
            return {
                "bbox": self.last_bbox,
                "path": self.car_path,
                "prob": float(prob)
            }

        row = best_idx // self.grid_dim
        col = best_idx % self.grid_dim
        
        dx, dy, w_norm, h_norm = 0, 0, 0.2, 0.2 # default small box
        
        if len(outputs_list) >= 3:
            sizes = np.array(outputs_list[1]).flatten()
            offsets = np.array(outputs_list[2]).flatten()
            w_norm = sizes[best_idx]
            h_norm = sizes[self.grid_dim**2 + best_idx]
            dx = offsets[best_idx]
            dy = offsets[self.grid_dim**2 + best_idx]

        cx_norm = (col + 0.5 + dx) / self.grid_dim
        cy_norm = (row + 0.5 + dy) / self.grid_dim
        
        cx_full = (cx - side/2) + cx_norm * side
        cy_full = (cy - side/2) + cy_norm * side
        w_full = w_norm * side
        h_full = h_norm * side
        
        new_bbox = [cx_full - w_full/2, cy_full - h_full/2, 
                    cx_full + w_full/2, cy_full + h_full/2]
        
        self.last_bbox = new_bbox
        self.car_path.append([cx_full, cy_full])
        
        return {
            "bbox": new_bbox,
            "path": self.car_path,
            "prob": float(prob)
        }

    def get_crop(self, img, cx, cy, side, output_size):
        x1, y1 = int(cx - side/2), int(cy - side/2)
        x2, y2 = int(cx + side/2), int(cy + side/2)
        h, w = img.shape[:2]
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return cv2.resize(crop, (output_size, output_size))

    def normalize(self, img):
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = (img - mean) / std
        return np.expand_dims(img, axis=0).astype(np.float32)
