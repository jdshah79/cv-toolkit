from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_version="yolov8n.pt"):
        """Initializes and loads the specified YOLO model."""
        self.model = YOLO(model_version)

    def detect(self, frame, confidence=0.5):
        """Processes a frame and returns structured bounding box data."""
        results = self.model(frame, stream=True)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf[0] >= confidence:
                    # Extract coordinates and class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = r.names[cls]
                    conf = float(box.conf[0])
                    
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "label": label,
                        "confidence": conf
                    })
        return detections
