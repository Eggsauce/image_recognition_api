from ultralytics import YOLO

# Create a new YOLO model from scratch (Load the model)
model = YOLO("yolo11n.yaml")

# Train the model using the 'coco8.yaml' dataset for 3 epochs (Use the model)
results = model.train(data="config.yaml", epochs=10000)
