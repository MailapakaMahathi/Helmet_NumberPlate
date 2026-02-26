from ultralytics import YOLO
import cv2

class HelmetDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"✅ Model loaded. Classes: {self.class_names}")

    def detect(self, frame):
        results = self.model(frame, conf=0.25)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detections.append({
                'class': self.class_names[cls_id],
                'conf': conf,
                'bbox': bbox
            })
        return detections

    def draw_boxes(self, frame, detections):
        for det in detections:
            class_name = det['class'].lower()
            x1, y1, x2, y2 = map(int, det['bbox'])

            # Color and label scheme
            if class_name == 'with helmet':
                color = (0, 255, 0)      # Green
                label = 'WITH HELMET'
            elif class_name == 'without helmet':
                color = (0, 0, 255)      # Red
                label = 'WITHOUT HELMET'
            elif class_name == 'number plate':
                color = (0, 255, 255)    # Yellow
                label = 'NUMBER PLATE'
            elif class_name == 'rider':
                color = (255, 165, 0)    # Orange
                label = 'RIDER'
            else:
                color = (255, 255, 255)
                label = class_name.upper()

            # Draw thick bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame,
                         (x1, y1 - text_size[1] - 10),
                         (x1 + text_size[0] + 10, y1),
                         color, -1)

            # Draw label text
            cv2.putText(frame, label,
                       (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 0), 2)

        return frame