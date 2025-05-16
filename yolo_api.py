import os
import cv2
import torch
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import tempfile

app = FastAPI()

# 1) Determine device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
assert device.startswith("cuda"), "CUDA not detected on this machine"

# 2) Load YOLOv8 (no device arg) and then move it
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)

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
    fname = video.filename.lower()
    if not fname.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Unsupported video format")

    # 3) Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as tmp:
        temp_path = tmp.name
        try:
            tmp.write(await video.read())
            tmp.flush()

            # 4) Frame-by-frame detection on GPU
            counts = Counter()
            for frame in extract_frames(temp_path):
                results = yolo_model(frame)  # runs on CUDA now
                for cls_id in results.boxes.cls:
                    label = yolo_model.names[int(cls_id)]
                    counts[label] += 1

            return {
                "device": device,
                "total_detections": sum(counts.values()),
                "counts": dict(counts)
            }
        finally:
            os.remove(temp_path)
