from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
DATA_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

model = YOLO(str(MODEL_PATH))
# model = YOLO(".\\best.pt")

results = model.train(
    data=str(DATA_CONFIG_PATH),
    project=str(RUNS_DIR),
    name="train",
    epochs=100,
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.4,
    scale=0.3,
    translate=0.1,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
)
