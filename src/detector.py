from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import numpy as np
import os
import logging
import base64
import io
import threading


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLO to run inference and return
    normalized and pixel coordinates along with class names and confidences.
    """
    # init runs once to get everything ready

    def __init__(self, weights_path: str, class_names_path: Optional[str] = None, device: Optional[str] = None):
        # Logger (debugging)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=os.environ.get(
                "YOLO_API_LOGLEVEL", "INFO"))
            # Logger (debugging) end
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at: {weights_path}")
        # Logger (debugging)
        self.logger.info(f"Loading YOLO weights from: {weights_path}")
        # Logger (debugging) end
        self.model = YOLO(weights_path)
        # Prefer names from model if available; fallback to obj.names file
        self.class_id_to_name = self._load_class_names(class_names_path)
        self.device = device  # let ultralytics auto-select if None
        self._inference_lock = threading.Lock()
        # Logger (debugging)
        self.logger.info(
            f"Detector initialized. device={self.device or 'auto'}, classes={len(self.class_id_to_name) if self.class_id_to_name else 'unknown'}")
        # Logger (debugging) end

    # looks inside the model or a text file to figure out what that number means (e.g., 0 = "login_button", 1 = "signup_button").
    def _load_class_names(self, class_names_path: Optional[str]) -> Dict[int, str]:
        # Try model-provided names first
        if hasattr(self.model, 'names') and isinstance(self.model.names, (list, dict)):
            if isinstance(self.model.names, dict):
                return {int(k): str(v) for k, v in self.model.names.items()}
            return {i: str(name) for i, name in enumerate(self.model.names)}

        # Fallback to obj.names file where each line is a class name
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
            return {i: name for i, name in enumerate(names)}

        # Final fallback
        return {}

    def _image_size(self, im: Image.Image) -> tuple[int, int]:
        return im.size  # (width, height)

    # crucial piece of math for YOLO models
    # translates absolute pixel coordinates into percentages.
    # YOLO naturally expects and outputs bounding boxes in a normalized format where values fall between 0 and 1. If the image is 1920 pixels wide, and a button starts at pixel 960, the normalized center is 0.5 (exactly 50% across the screen).
    # calculates the center by adding half the width to the starting point (x_center = x_min + w / 2.0), and then divides that by the total image width (x_center / img_w) to get that percentage. Predictions will still be accurate even if you shrink or enlarge the image
    def _to_normalized(self, xyxy: np.ndarray, img_w: int, img_h: int) -> Dict[str, float]:
        x_min, y_min, x_max, y_max = xyxy.tolist()
        w = x_max - x_min
        h = y_max - y_min
        x_center = x_min + w / 2.0
        y_center = y_min + h / 2.0
        return {
            "x_center": x_center / img_w,
            "y_center": y_center / img_h,
            "width": w / img_w,
            "height": h / img_h,
        }

    def _load_image(self, image_path: Optional[str] = None, image_b64: Optional[str] = None) -> Image.Image:
        """Load a PIL Image from either a file path or a base64 string."""
        if image_b64 is not None:
            try:
                image_bytes = base64.b64decode(image_b64, validate=True)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to decode base64 image: {e}")

        if image_path is not None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            return Image.open(image_path).convert('RGB')

        raise ValueError("Either image_path or image_b64 must be provided")

    def _run_detection(
        self,
        image: Image.Image,
        image_source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """Core detection logic shared by both detect_on_path and detect_on_b64."""
        img_w, img_h = self._image_size(image)

        predict_kwargs = {"conf": conf,
                          "device": self.device, "verbose": False, "iou": iou}
        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz

        self.logger.info(
            f"Detect source={image_source}, conf={conf}, device={self.device or 'auto'}")
        with self._inference_lock:
            results = self.model.predict(source=np.array(image), **predict_kwargs)

        num_results = 0 if results is None else len(results)
        self.logger.debug(f"model.predict results={num_results}")

        detections: List[Dict[str, Any]] = []

        if not results:
            self.logger.warning("No results returned by model.predict")
            debug_info = {
                "num_results": 0,
                "num_boxes": 0,
                "conf_threshold": conf,
                "device": self.device or "auto",
                "classes_loaded": list(self.class_id_to_name.values()) if self.class_id_to_name else None
            } if return_raw else None
            return {
                "image_source": image_source,
                "image_width": img_w,
                "image_height": img_h,
                "detections": detections,
                "debug": debug_info
            }

        result = results[0]
        boxes = result.boxes

        try:
            num_boxes = 0 if boxes is None else int(
                getattr(boxes.xyxy, 'shape', [0])[0])
        except Exception:
            try:
                num_boxes = len(boxes)
            except Exception:
                num_boxes = 0
        self.logger.info(f"Boxes found={num_boxes}")

        xyxy_all = boxes.xyxy.cpu().numpy()
        conf_all = boxes.conf.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy()

        for i in range(len(xyxy_all)):
            xyxy = xyxy_all[i]
            conf_score = float(conf_all[i])
            cls_id = int(cls_all[i])
            cls_name = self.class_id_to_name.get(cls_id, str(cls_id))

            x_min, y_min, x_max, y_max = [round(v) for v in xyxy.tolist()]
            norm = self._to_normalized(xyxy, img_w, img_h)

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf_score,
                "bbox_pixel": {
                    "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                    "width": x_max - x_min, "height": y_max - y_min,
                    "x_center": int((x_min + x_max) / 2), "y_center": int((y_min + y_max) / 2)
                },
                "bbox_normalized": {
                    "x_center": round(norm["x_center"], 6),
                    "y_center": round(norm["y_center"], 6),
                    "width": round(norm["width"], 6),
                    "height": round(norm["height"], 6)
                },
            })

        debug_info = None
        if return_raw:
            try:
                xyxy_sample = boxes.xyxy[:5].cpu().numpy(
                ).tolist() if hasattr(boxes, 'xyxy') else []
            except Exception:
                xyxy_sample = []
            try:
                conf_sample = boxes.conf[:5].cpu().numpy(
                ).tolist() if hasattr(boxes, 'conf') else []
            except Exception:
                conf_sample = []
            try:
                cls_sample = boxes.cls[:5].cpu().numpy(
                ).tolist() if hasattr(boxes, 'cls') else []
            except Exception:
                cls_sample = []

            debug_info = {
                "num_results": num_results,
                "num_boxes": len(detections),
                "conf_threshold": conf,
                "iou_threshold": iou,
                "imgsz": imgsz,
                "device": self.device or "auto",
                "classes_loaded": list(self.class_id_to_name.values()) if self.class_id_to_name else None,
                "orig_shape": getattr(result, 'orig_shape', None),
                "model_names": getattr(self.model, 'names', None),
                "model_nc": getattr(self.model, 'nc', None),
                "raw_boxes_xyxy_sample": xyxy_sample,
                "raw_boxes_conf_sample": conf_sample,
                "raw_boxes_cls_sample": cls_sample
            }

        return {
            "image_source": image_source,
            "image_width": img_w,
            "image_height": img_h,
            "detections": detections,
            "debug": debug_info
        }

    # Debugging (iou, imgsz) iou =intersection over union.
    def detect_on_image(self, image_path: str, conf: float = 0.25, iou: float = 0.45, imgsz: Optional[int] = None, return_raw: bool = False) -> Dict[str, Any]:
        image = self._load_image(image_path=image_path)
        return self._run_detection(image, image_source=image_path, conf=conf, iou=iou, imgsz=imgsz, return_raw=return_raw)

    def detect_on_b64(self, image_b64: str, conf: float = 0.25, iou: float = 0.45, imgsz: Optional[int] = None, return_raw: bool = False) -> Dict[str, Any]:
        image = self._load_image(image_b64=image_b64)
        return self._run_detection(image, image_source="base64", conf=conf, iou=iou, imgsz=imgsz, return_raw=return_raw)
