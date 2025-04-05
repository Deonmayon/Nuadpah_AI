from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("model/new_best_seg.pt")

# Export the model to TFLite format
model.export(format="tflite")  # creates 'yolo11n_float32.tflite'

# Load the exported TFLite model
tflite_model = YOLO("new_best_seg.tflite")