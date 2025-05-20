import os
import cv2
import torch
import logging
import time
import requests
import tempfile
from collections import Counter
from fastapi import FastAPI, HTTPException, Body
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yolo_api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if not device.startswith("cuda"):
    logger.warning("CUDA not detected—running on CPU")

# Load YOLOv8 model once
logger.info("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)
logger.info("YOLOv8 model loaded successfully")


def extract_frames(video_path: str):
    """Yield one frame per second from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    for sec in range(duration + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)  # seek to ms
        ret, frame = cap.read()
        if not ret:
            break
        logger.info(f"Extracted frame at {sec}s")
        yield frame

    cap.release()


async def run_detection_on_path(video_path: str):
    """Run YOLO detection over all frames and aggregate counts."""
    counts = Counter()
    frame_count = 0
    start = time.time()
    for frame in extract_frames(video_path):
        frame_count += 1
        results = yolo_model(frame)
        for res in results:
            if res.boxes is not None:
                for cls_id in res.boxes.cls:
                    label = yolo_model.names[int(cls_id)]
                    counts[label] += 1
    elapsed = time.time() - start
    return {
        "device": device,
        "total_frames": frame_count,
        "total_detections": sum(counts.values()),
        "counts": dict(counts),
        "processing_time": f"{elapsed:.2f}s"
    }


@app.post("/detect_objects/")
async def detect_objects_url(video_url: str = Body(..., embed=True)):
    """
    URL-based endpoint: download the video and run object detection over every frame.
    Request body:
      { "video_url": "https://example.com/video.mp4" }
    """
    logger.info(f"Received URL for detection: {video_url}")

    # Validate extension
    ext = os.path.splitext(video_url)[1].lower()
    if ext not in (".mp4", ".avi", ".mov"):
        raise HTTPException(400, "Unsupported video format")

    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        path = tmp.name
        resp = requests.get(video_url, stream=True)
        if resp.status_code != 200:
            logger.error(f"Failed to download video: HTTP {resp.status_code}")
            raise HTTPException(400, "Failed to download video")
        for chunk in resp.iter_content(8192):
            tmp.write(chunk)
        tmp.flush()

    try:
        # Run YOLO detection on the downloaded file
        return await run_detection_on_path(path)
    finally:
        # Clean up
        try:
            os.remove(path)
        except OSError:
            logger.warning(f"Could not delete temp file: {path}")
