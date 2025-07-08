from ultralytics import YOLO
from PIL import Image
from typing import List, Tuple

# Load trained YOLOv8 once
_yolo = YOLO("models/yolov8_pin.pt")


def detect_boxes(img: Image.Image, conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
    """Return list of (x1,y1,x2,y2) boxes"""
    results = _yolo.predict(img, conf=conf, verbose=False)
    b = results[0].boxes.xyxy.cpu().numpy()
    return [tuple(map(int, box[:4])) for box in b]