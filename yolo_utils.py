import cv2
import numpy as np
from ultralytics import YOLO

class DynamicMasker:
    def __init__(self):
        print("Loading YOLOv8n for Dynamic Masking...")
        self.model = YOLO('yolov8n.pt') 
        self.dynamic_classes = [0, 1, 2, 3, 5, 7]

    def get_static_mask(self, image_bgr):
        height, width = image_bgr.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        results = self.model(image_bgr, verbose=False, classes=self.dynamic_classes)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1) 
                
        return mask