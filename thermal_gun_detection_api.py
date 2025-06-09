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

def cleanup_files(request_id: str, temp_file_path: str = None, work_dir: str = None):
    """Clean up temporary files"""
    try:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"[{request_id}] Cleaned up temp file")
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to remove temp file: {e}")

    try:
        if work_dir and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            logger.info(f"[{request_id}] Cleaned up work directory")
    except Exception as e:
        logger.warning(f"[{request_id}] Failed to remove work directory: {e}")

@app.post("/detect_thermal_guns_image/")
async def detect_thermal_guns_in_image(
    image_url: str = Body(..., embed=True),
    confidence_threshold: float = Body(0.25, embed=True)
):
    """Detect thermal guns in a single image using trained YOLO model"""

    if model is None:
        raise HTTPException(500, "Model not loaded")

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting thermal gun detection in image: {image_url}")

    # Validate image URL
    parsed_url = urlparse(image_url)
    ext = Path(parsed_url.path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        raise HTTPException(400, f"Unsupported image format: '{ext}'. Supported: .jpg, .jpeg, .png, .bmp, .tiff, .webp")

    temp_image_path = None

    try:
        # Create temporary image file
        temp_image_fd, temp_image_path = tempfile.mkstemp(suffix=ext, prefix=f"image_{request_id}_")

        try:
            # Download image
            logger.info(f"[{request_id}] Downloading image...")
            r = requests.get(image_url, stream=True, timeout=30)
            r.raise_for_status()

            with os.fdopen(temp_image_fd, 'wb') as temp_file:
                for chunk in r.iter_content(8192):
                    temp_file.write(chunk)

        except Exception as e:
            raise HTTPException(400, f"Failed to download image: {str(e)}")

        logger.info(f"[{request_id}] Image downloaded successfully")

        # Run detection on the image
        logger.info(f"[{request_id}] Running thermal gun detection...")
        detections = detect_guns_in_frame(temp_image_path, confidence_threshold)

        # Prepare response
        response_data = {
            "request_id": request_id,
            "confidence_threshold": confidence_threshold,
            "summary": {
                "total_gun_detections": len(detections),
                "guns_found": len(detections) > 0
            },
            "detections": detections
        }

        logger.info(f"[{request_id}] Detection complete. Found {len(detections)} guns")
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error during processing: {e}")
        raise HTTPException(500, f"Detection failed: {str(e)}")

    finally:
        # Cleanup
        logger.info(f"[{request_id}] Cleaning up...")
        cleanup_files(request_id, temp_image_path)

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
        gun_detection_times = []
        total_gun_count = 0
        detailed_detections = []

        for frame_file, second in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)

            # Detect guns in this frame
            detections = detect_guns_in_frame(frame_path, confidence_threshold)

            if len(detections) > 0:
                gun_detection_times.append({
                    "time_seconds": second,
                    "gun_count": len(detections),
                    "max_confidence": max([d["confidence"] for d in detections]),
                    "detections": detections
                })
                total_gun_count += len(detections)

            # Store all frame results for detailed analysis if needed
            detailed_detections.append({
                "time_seconds": second,
                "gun_count": len(detections),
                "detections": detections
            })

        # Calculate summary
        frames_with_guns = len(gun_detection_times)
        detection_rate = (frames_with_guns / len(frame_files)) * 100 if frame_files else 0

        logger.info(f"[{request_id}] Detection complete. Found guns in {frames_with_guns}/{len(frame_files)} frames")

        # Prepare focused response
        response_data = {
            "request_id": request_id,
            "confidence_threshold": confidence_threshold,
            "video_info": {
                "duration_seconds": round(duration, 2),
                "total_frames_analyzed": len(frame_files)
            },
            "gun_detection_summary": {
                "guns_found": len(gun_detection_times) > 0,
                "total_gun_detections": total_gun_count,
                "frames_with_guns": frames_with_guns,
                "detection_rate_percent": round(detection_rate, 1),
                "detection_times": [t["time_seconds"] for t in gun_detection_times]
            },
            "gun_detection_details": gun_detection_times
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
