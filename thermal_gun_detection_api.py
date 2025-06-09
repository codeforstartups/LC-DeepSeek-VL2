import os
import cv2
import asyncio
import tempfile
import logging
import requests
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from ultralytics import YOLO
from urllib.parse import urlparse
import torch

app = FastAPI(title="Thermal Gun Detection API", version="1.0.0")
logger = logging.getLogger("thermal_gun_detection")
logging.basicConfig(level=logging.INFO)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load the trained thermal gun detection model on startup"""
    global model

    # Model path - MyTrainedModels folder is on same level as this file
    model_path = "MyTrainedModels/ThermalPistol_v2_improved/weights/best.pt"

    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}")
        raise RuntimeError("Model file not found")

    try:
        logger.info(f"Loading thermal gun detection model from: {model_path}")
        model = YOLO(model_path)
        logger.info("✅ Thermal gun detection model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError("Model loading failed")

def extract_frames_1fps(video_path: str, out_dir: str):
    """Extract frames at 1 FPS from video"""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video FPS: {fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")

    frame_files = []
    second = 0

    while True:
        # Set position to the next second
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * second)
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = f"frame_{second:03d}.jpg"
        frame_path = os.path.join(out_dir, frame_file)
        cv2.imwrite(frame_path, frame)
        frame_files.append((frame_file, second))
        second += 1

    cap.release()
    return frame_files, duration

def detect_guns_in_frame(frame_path: str, confidence_threshold: float = 0.25):
    """Run thermal gun detection on a single frame"""
    try:
        # Run inference
        results = model(frame_path, conf=confidence_threshold, verbose=False)

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i].cpu().numpy())

                detection = {
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    },
                    "confidence": round(confidence, 3),
                    "class": "thermal_gun"
                }
                detections.append(detection)

        return detections

    except Exception as e:
        logger.error(f"Error detecting guns in frame {frame_path}: {e}")
        return []

def cleanup_files(request_id: str, temp_video_path: str = None, work_dir: str = None):
    """Clean up temporary files"""
    try:
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            logger.info(f"[{request_id}] Cleaned up video file")
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to remove video file: {e}")

    try:
        if work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            logger.info(f"[{request_id}] Cleaned up work directory")
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to remove work directory: {e}")

@app.post("/detect_thermal_guns/")
async def detect_thermal_guns_in_video(
    video_url: str = Body(..., embed=True),
    confidence_threshold: float = Body(0.25, embed=True)
):
    """Detect thermal guns in video frames using trained YOLO model"""

    if model is None:
        raise HTTPException(500, "Model not loaded")

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting thermal gun detection in video: {video_url}")

    # Validate video URL
    parsed_url = urlparse(video_url)
    ext = Path(parsed_url.path).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, f"Unsupported video format: '{ext}'. Supported: .mp4, .avi, .mov, .mkv")

    temp_video_path = None
    work_dir = None

    try:
        # Create temporary directories
        work_dir = tempfile.mkdtemp(prefix=f"thermal_detection_{request_id}_")
        frame_dir = os.path.join(work_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # Download video
        logger.info(f"[{request_id}] Downloading video...")
        temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=ext, prefix=f"video_{request_id}_")

        try:
            r = requests.get(video_url, stream=True, timeout=60)
            r.raise_for_status()

            with os.fdopen(temp_video_fd, 'wb') as temp_file:
                for chunk in r.iter_content(8192):
                    temp_file.write(chunk)

        except Exception as e:
            raise HTTPException(400, f"Failed to download video: {str(e)}")

        logger.info(f"[{request_id}] Video downloaded successfully")

        # Extract frames
        logger.info(f"[{request_id}] Extracting frames at 1 FPS...")
        frame_files, duration = extract_frames_1fps(temp_video_path, frame_dir)
        logger.info(f"[{request_id}] Extracted {len(frame_files)} frames")

        # Process each frame
        logger.info(f"[{request_id}] Running thermal gun detection...")
        all_detections = []
        frames_with_guns = []
        total_gun_count = 0

        for frame_file, second in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)

            # Detect guns in this frame
            detections = detect_guns_in_frame(frame_path, confidence_threshold)

            frame_result = {
                "second": second,
                "frame_file": frame_file,
                "gun_count": len(detections),
                "detections": detections
            }

            all_detections.append(frame_result)
            total_gun_count += len(detections)

            if len(detections) > 0:
                frames_with_guns.append({
                    "second": second,
                    "gun_count": len(detections),
                    "max_confidence": max([d["confidence"] for d in detections])
                })

        # Calculate summary
        frames_with_detections = len(frames_with_guns)
        detection_rate = (frames_with_detections / len(frame_files)) * 100 if frame_files else 0

        logger.info(f"[{request_id}] Detection complete. Found guns in {frames_with_detections}/{len(frame_files)} frames")

        # Prepare response
        response_data = {
            "request_id": request_id,
            "video_url": video_url,
            "video_duration_seconds": round(duration, 2),
            "total_frames_analyzed": len(frame_files),
            "confidence_threshold": confidence_threshold,
            "summary": {
                "total_gun_detections": total_gun_count,
                "frames_with_guns": frames_with_detections,
                "detection_rate_percent": round(detection_rate, 1),
                "seconds_with_guns": [f["second"] for f in frames_with_guns]
            },
            "frames_with_guns": frames_with_guns,
            "all_frame_results": all_detections
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error during processing: {e}")
        raise HTTPException(500, f"Detection failed: {str(e)}")

    finally:
        # Cleanup
        logger.info(f"[{request_id}] Cleaning up...")
        cleanup_files(request_id, temp_video_path, work_dir)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_info": "Thermal Gun Detection YOLOv8" if model else "No model loaded"
    }

@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    global model

    if model is None:
        raise HTTPException(500, "Model not loaded")

    return {
        "model_type": "YOLOv8 Thermal Gun Detection",
        "classes": ["thermal_gun"],
        "input_size": 640,
        "framework": "Ultralytics YOLO"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
