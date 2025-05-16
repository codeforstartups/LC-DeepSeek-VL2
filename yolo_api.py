import os
import cv2
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import torch
import tempfile

app = FastAPI()

# Check GPU availability and load model accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt", device=device)

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

    # Create temp file in system temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as temp_file:
        temp_path = temp_file.name
        try:
            # Save upload
            content = await video.read()
            temp_file.write(content)
            temp_file.flush()

            # Run detection on each frame
            counts = Counter()
            for frame in extract_frames(temp_path):
                results = yolo_model(frame)
                for cls_id in results.boxes.cls:
                    label = yolo_model.names[int(cls_id)]
                    counts[label] += 1

            return {"total_detections": sum(counts.values()), "counts": dict(counts)}
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
