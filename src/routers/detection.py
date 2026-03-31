from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from detector import YoloDetector
from models.api import DetectionImageData
import os

router = APIRouter(prefix="/detection", tags=["detection"])


@router.post("/image-data")
def detect_from_image_data(
    req: DetectionImageData,
    target_class_name: Optional[str] = Query(
        default=None,
        description="Optional class name to filter detections (e.g. mainpage_login_button)",
    ),
):
    weights_path = os.getenv("WEIGHTS_PATH")
    classname_path = os.getenv("CLASSNAME_PATH")
    try:
        detector = YoloDetector(weights_path=weights_path,
                                class_names_path=classname_path, device=req.device)
        result = detector.detect_on_b64(
            image_b64=req.image_data,
            conf=req.conf,
            iou=req.iou,
            imgsz=req.imgsz,
            return_raw=req.debug,
        )

        detections = result.get("detections", [])

        # Prefer explicit query parameter if provided; fall back to body field
        class_filter = target_class_name or req.class_name_path
        if class_filter is not None:
            detections = [
                d for d in detections if d.get("class_name") == class_filter
            ]
        if req.min_score is not None:
            detections = [d for d in detections if d.get(
                "confidence", 0.0) >= float(req.min_score)]
        if req.return_top and detections:
            detections = [
                max(detections, key=lambda d: d.get("confidence", 0.0))]

        result["detections"] = detections
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Detection error: {str(e)}")
