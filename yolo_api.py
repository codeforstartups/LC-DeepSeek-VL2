import os
import cv2
import torch
import logging
import time
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
import tempfile

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

# 1) Determine device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
assert device.startswith("cuda"), "CUDA not detected on this machine"

# 2) Load YOLOv8 (no device arg) and then move it
logger.info("Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
yolo_model.to(device)
logger.info("YOLOv8 model loaded successfully")

def extract_frames(video_path: str):
    logger.info(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        raise RuntimeError("Cannot open video file")

    frame_count = 0
    ret, frame = cap.read()
    while ret:
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames")
        yield frame
        ret, frame = cap.read()

    logger.info(f"Total frames processed: {frame_count}")
    cap.release()

@app.post("/detect_objects/")
async def detect_objects(video: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received video upload: {video.filename}")

    fname = video.filename.lower()
    if not fname.endswith((".mp4", ".avi", ".mov")):
        logger.error(f"Unsupported video format: {fname}")
        raise HTTPException(400, "Unsupported video format")

    # 3) Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(fname)[1]) as tmp:
        temp_path = tmp.name
        try:
            logger.info(f"Saving video to temporary file: {temp_path}")
            tmp.write(await video.read())
            tmp.flush()

            # 4) Frame-by-frame detection on GPU
            logger.info("Starting object detection...")
            counts = Counter()
            frame_count = 0
            detection_start = time.time()

            for frame in extract_frames(temp_path):
                frame_count += 1
                results = yolo_model(frame)  # runs on CUDA now
                # Process each result in the list
                for result in results:
                    if result.boxes is not None:  # Check if boxes exist
                        for cls_id in result.boxes.cls:
                            label = yolo_model.names[int(cls_id)]
                            counts[label] += 1

            detection_time = time.time() - detection_start
            logger.info(f"Detection completed in {detection_time:.2f} seconds")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Detection results: {dict(counts)}")

            return {
                "device": device,
                "total_detections": sum(counts.values()),
                "counts": dict(counts),
                "processing_time": f"{detection_time:.2f} seconds",
                "total_frames": frame_count
            }
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise HTTPException(500, f"Error processing video: {str(e)}")
        finally:
            logger.info("Cleaning up temporary file")
            os.remove(temp_path)

    total_time = time.time() - start_time
    logger.info(f"Total request processing time: {total_time:.2f} seconds")
