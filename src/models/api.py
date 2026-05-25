from typing import Annotated, Optional
from pydantic import BaseModel, StringConstraints

# Largest training image (1920x1080) is ~4.9MB; base64 adds ~33% → ~6.5MB.
# 10MB gives comfortable headroom while bounding runaway payloads.
_MAX_IMAGE_B64 = 10_000_000

class DetectionImageData(BaseModel):
    image_data: Annotated[str, StringConstraints(max_length=_MAX_IMAGE_B64)]
    class_name_path: Optional[str] = None  # target_element
    conf: float = 0.05
    iou: float = 0.45
    imgsz: Optional[int] = None
    device: Optional[str] = None
    debug: bool = False
    min_score: Optional[float] = None
    return_top: bool = False
