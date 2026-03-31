from typing import Optional
from pydantic import BaseModel


class DetectionImageData(BaseModel):
    image_data: str
    class_name_path: Optional[str] = None  # target_element
    conf: float = 0.05
    iou: float = 0.45
    imgsz: Optional[int] = None
    device: Optional[str] = None
    debug: bool = False
    min_score: Optional[float] = None
    return_top: bool = False
