from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from detector import YoloDetector
from models.api import DetectionImageData
import os
import asyncio
import threading
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detection", tags=["detection"])

# Thread-safe global cache for YoloDetector instances
# Key: (weights_path, classname_path, device)
_detector_cache = {}
_cache_lock = threading.Lock()

def get_detector(weights_path: str, classname_path: Optional[str], device: Optional[str]) -> YoloDetector:
    key = (weights_path, classname_path, device)
    with _cache_lock:
        if key not in _detector_cache:
            _detector_cache[key] = YoloDetector(
                weights_path=weights_path,
                class_names_path=classname_path,
                device=device
            )
        return _detector_cache[key]


@router.post("/image-data")
async def detect_from_image_data(
    req: DetectionImageData,
    target_class_name: Optional[str] = Query(
        default=None,
        description="Optional class name to filter detections (e.g. mainpage_login_button)",
    ),
):
    weights_path = os.getenv("WEIGHTS_PATH")
    classname_path = os.getenv("CLASSNAME_PATH")
    
    if not weights_path:
        raise HTTPException(
            status_code=400,
            detail="WEIGHTS_PATH environment variable is not set"
        )
        
    try:
        # Load or retrieve the detector asynchronously in a separate thread
        detector = await asyncio.to_thread(
            get_detector,
            weights_path,
            classname_path,
            req.device
        )
        # Run model inference in a worker thread to keep FastAPI non-blocking
        result = await asyncio.to_thread(
            detector.detect_on_b64,
            image_b64=req.image_data,
            conf=req.conf,
            iou=req.iou,
            imgsz=req.imgsz,
            return_raw=req.debug,
        )

        detections = result.get("detections", [])

        # Prefer explicit query parameter if provided; fall back to body field
        class_filter = target_class_name or req.class_name_path

        # If user asks for one specific class but does not pass min_score,
        # use a safer default to reduce low-confidence false positives.
        effective_min_score = req.min_score
        if class_filter is not None and effective_min_score is None:
            effective_min_score = 0.25

        if class_filter is not None:
            detections = [
                d for d in detections if d.get("class_name") == class_filter
            ]
        if effective_min_score is not None:
            detections = [d for d in detections if d.get(
                "confidence", 0.0) >= float(effective_min_score)]

        # Keep only one detection per class_name (highest confidence)
        if detections:
            best_by_class = {}
            for d in detections:
                class_name = d.get("class_name")
                current_best = best_by_class.get(class_name)
                if (
                    current_best is None
                    or d.get("confidence", 0.0) > current_best.get("confidence", 0.0)
                ):
                    best_by_class[class_name] = d
            detections = list(best_by_class.values())

        if req.return_top and detections:
            detections = [
                max(detections, key=lambda d: d.get("confidence", 0.0))]

        result["detections"] = detections
        if class_filter is not None and not detections:
            result["message"] = (
                f'No "{class_filter}" detected in image with current thresholds.'
            )
        return result
    except FileNotFoundError as e:
        logger.error("Resource not found: %s", e)
        raise HTTPException(status_code=404, detail="Model or resource file not found.")
    except RuntimeError as e:
        logger.error("Runtime error during detection: %s", e)
        raise HTTPException(status_code=500, detail="Detection failed due to a runtime error.")
    except Exception as e:
        logger.exception("Unexpected error during detection: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during detection.")
