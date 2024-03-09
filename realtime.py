import cv2
import torch
import numpy as np

def load_yolov5_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def detect_cracks_yolo(frame, model):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Perform dilation and erosion to close gaps in between object edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    # Find contours in the edges image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes and labels on the frame
    results = model(frame)
    annotated_frame = results.render()[0]
    
    return annotated_frame

# Path to YOLOv5 model weights
weights_path = "/Users/boomika/Desktop/kaviya/yolov5/runs/train/exp/weights/best.pt"

# Load YOLOv5 model
model = load_yolov5_model(weights_path)

# Initialize video capture from file
cap = cv2.VideoCapture(r"/Users/boomika/Downloads/WhatsApp Video 2024-03-09 at 12.24.17.mp4")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect cracks using YOLOv5 and annotate the frame
    annotated_frame = detect_cracks_yolo(frame, model)
    
    # Display result
    cv2.imshow('Crack Detection', annotated_frame)
    
    # Check for 'q' key pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
