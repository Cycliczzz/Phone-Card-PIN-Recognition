#!/usr/bin/env bash
# Train YOLOv8 on scratch‑card pins
yolo detect train \
  data=config/pin.yaml \
  model=yolov8n.pt \
  epochs=50 \
  imgsz=640 \
  project=runs/detect \
  name=train