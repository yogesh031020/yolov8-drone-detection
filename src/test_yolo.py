from ultralytics import YOLO
import cv2
import torch

print("Loading YOLOv8 model...")
print("GPU:", torch.cuda.get_device_name(0))

# Load pretrained YOLOv8 model
# This will auto download yolov8n.pt (6MB) on first run
model = YOLO("yolov8n.pt")
model.to("cuda")
print("Model loaded on GPU!")

# Open webcam (0 = default camera)
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not found! Trying video file...")
    cap = cv2.VideoCapture("test_video.mp4")

print("Running detection - Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received!")
        break

    # Run YOLOv8 detection
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated = results[0].plot()

    # Show FPS and detection count
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    cv2.putText(annotated,
        "Objects detected: " + str(count),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user!")
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")