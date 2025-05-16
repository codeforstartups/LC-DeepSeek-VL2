import os
import cv2
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import torch

app = FastAPI()

# Ensure CUDA is used
assert torch.cuda.is_available(), "CUDA not detected on this machine"

# Load YOLOv8 on GPU at startup
yolo_model = YOLO("yolov8n.pt", device="cuda:0")

def extract_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")
    ret, frame = cap.read()
    while ret:
        yield frame
        ret, frame = cap.read()
    cap.release()

@app.post("/detect_objects/")
async def detect_objects(video: UploadFile = File(...)):
    # Validate format
    fname = video.filename.lower()
    if not fname.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Unsupported video format")

    temp_path = f"/tmp/{video.filename}"
    try:
        # Save upload
        with open(temp_path, "wb") as f:
            f.write(await video.read())

        # Run detection on each frame
        counts = Counter()
        for frame in extract_frames(temp_path):
            # YOLO inference on GPU
            results = yolo_model(frame)
            for cls_id in results.boxes.cls:
                label = yolo_model.names[int(cls_id)]
                counts[label] += 1

        return {"total_detections": sum(counts.values()), "counts": dict(counts)}
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
