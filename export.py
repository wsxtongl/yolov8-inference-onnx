from ultralytics import YOLO
model = YOLO("yolov8s.pt")
success = model.export(format="onnx", half=False, dynamic=True, opset=17)