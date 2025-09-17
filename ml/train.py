from ultralytics import YOLO

# Load the model
model = YOLO("yolo11n.pt")

# Train the model
results = model.train(data="data/data.yaml", epochs=100, imgsz=640)