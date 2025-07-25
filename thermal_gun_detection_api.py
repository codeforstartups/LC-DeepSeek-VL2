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

# pytube2 import with fallback
try:
    from pytube import YouTube
    PYTUBE_AVAILABLE = True
except ImportError:
    PYTUBE_AVAILABLE = False
    logging.warning("⚠️ pytube2 not installed. YouTube URLs won't work. Install with: pip install pytube2")

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

def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_domains = [
        'youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com'
    ]

    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace('www.', '')

    return any(youtube_domain in domain for youtube_domain in youtube_domains)

def download_youtube_video(video_url: str, output_path: str, request_id: str):
    """Download YouTube video using pytube2"""
    if not PYTUBE_AVAILABLE:
        raise HTTPException(400, "pytube2 not available. Cannot download YouTube URLs.")

    logger.info(f"[{request_id}] Downloading YouTube video with pytube2...")

    try:
        # Create YouTube object
        yt = YouTube(video_url)

        # Try to get basic info first
        try:
            title = yt.title or "Unknown Title"
            length = yt.length or 0
            views = yt.views or 0
        except Exception as info_error:
            logger.warning(f"[{request_id}] Could not get video info: {info_error}")
            title = "Unknown Title"
            length = 0
            views = 0

        logger.info(f"[{request_id}] YouTube Video: {title}")
        logger.info(f"[{request_id}] Duration: {length} seconds")
        logger.info(f"[{request_id}] Views: {views}")

        # Get best progressive stream (video + audio in one file)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()

        if not stream:
            # Fallback: get best available stream
            stream = yt.streams.get_highest_resolution()

        if not stream:
            # Second fallback: get any video stream
            stream = yt.streams.filter(file_extension='mp4').first()

        if not stream:
            raise HTTPException(400, "No suitable video stream found for this YouTube video")

        logger.info(f"[{request_id}] Selected quality: {stream.resolution} - {stream.mime_type}")

        # Download to specific path
        downloaded_file = stream.download(output_path=os.path.dirname(output_path),
                                        filename=os.path.basename(output_path))

        logger.info(f"[{request_id}] YouTube video downloaded successfully with pytube2")
        return downloaded_file

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] pytube2 error: {error_msg}")

        # Provide helpful error messages
        if "Exception while accessing title" in error_msg:
            raise HTTPException(400, f"YouTube video access failed - Video may be private, restricted, or unavailable. Try a different YouTube URL.")
        elif "HTTP Error 403" in error_msg:
            raise HTTPException(400, f"YouTube blocked access - Video may be age-restricted or region-locked.")
        elif "No suitable video stream" in error_msg:
            raise HTTPException(400, f"No downloadable streams found for this YouTube video.")
        else:
            raise HTTPException(400, f"Failed to download YouTube video: {error_msg}")

def download_direct_video(video_url: str, temp_video_path: str, request_id: str):
    """Download direct video URL using requests (existing logic)"""
    logger.info(f"[{request_id}] Downloading direct video with requests...")

    try:
        r = requests.get(video_url, stream=True, timeout=60)
        r.raise_for_status()

        with open(temp_video_path, 'wb') as temp_file:
            for chunk in r.iter_content(8192):
                temp_file.write(chunk)

        logger.info(f"[{request_id}] Direct video downloaded successfully")

    except Exception as e:
        raise HTTPException(400, f"Failed to download direct video: {str(e)}")

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

                # Only return detections with confidence >= 70%
                if confidence >= 0.70:
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
    """Detect thermal guns in video frames using trained YOLO model

    Supports:
    - Direct video URLs (.mp4, .avi, .mov, .mkv)
    - YouTube URLs (requires pytube2)
    """

    if model is None:
        raise HTTPException(500, "Model not loaded")

    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Starting thermal gun detection in video: {video_url}")

    # Check if it's a YouTube URL
    is_youtube = is_youtube_url(video_url)

    # Validate URL format
    if not is_youtube:
        # For direct URLs, validate file extension
        parsed_url = urlparse(video_url)
        ext = Path(parsed_url.path).suffix.lower()
        if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise HTTPException(400, f"Unsupported video format: '{ext}'. Supported: .mp4, .avi, .mov, .mkv for direct URLs, or YouTube URLs")
    else:
        # For YouTube URLs, use .mp4 as default extension
        ext = ".mp4"

    temp_video_path = None
    work_dir = None

    try:
        # Create temporary directories
        work_dir = tempfile.mkdtemp(prefix=f"thermal_detection_{request_id}_")
        frame_dir = os.path.join(work_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        # Create temporary video file path
        temp_video_path = os.path.join(work_dir, f"video_{request_id}{ext}")

        # Download video using appropriate method
        logger.info(f"[{request_id}] Downloading video...")

        if is_youtube:
            # Use pytube2 for YouTube URLs
            actual_video_path = download_youtube_video(video_url, temp_video_path, request_id)
            # pytube2 might change the filename, so update our path
            temp_video_path = actual_video_path
        else:
            # Use requests for direct URLs
            download_direct_video(video_url, temp_video_path, request_id)

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

    # Respect environment variable for device
    device = os.getenv("TORCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "model_info": "Thermal Gun Detection YOLOv8" if model else "No model loaded",
        "pytube2_available": PYTUBE_AVAILABLE,
        "download_methods": {
            "youtube_urls": "pytube2" if PYTUBE_AVAILABLE else "not available",
            "direct_urls": "requests"
        },
        "supported_sources": [
            "Direct video URLs (.mp4, .avi, .mov, .mkv)",
            "YouTube URLs" if PYTUBE_AVAILABLE else "YouTube URLs (install pytube2)"
        ]
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
