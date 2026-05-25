from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="config.yaml",
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
