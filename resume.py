from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train17/weights/last.pt")  # load a pretrained model (recommended for training)
# Train the model
model.train(resume=True)
