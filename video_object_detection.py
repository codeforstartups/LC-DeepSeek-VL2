import os
import cv2
import httpx
import asyncio
import tempfile
import logging
import requests
import math
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from httpx import HTTPStatusError, TimeoutException
from urllib.parse import urlparse

app = FastAPI(title="Video Object Detection API", version="1.0.0")
logger = logging.getLogger("video_object_detection")
logging.basicConfig(level=logging.INFO)

# API URLs
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://localhost:8000/analyze/")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

# Log configuration on startup
logger.info(f"DeepSeek API URL: {DEEPSEEK_API_URL}")
logger.info(f"Ollama Chat URL: {OLLAMA_CHAT_URL}")

client = httpx.AsyncClient(timeout=None)

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

def extract_1fps(video_path: str, out_dir: str):
    """Extract frames at 1 FPS from video"""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Video FPS: {fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")

    idx = 0
    frame_files = []

    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_file = f"frame_{idx:03d}.jpg"
        frame_path = os.path.join(out_dir, frame_file)
        cv2.imwrite(frame_path, frame)
        frame_files.append(frame_file)
        idx += 1

    cap.release()
    return sorted(frame_files), duration

def make_panoramas_6_frames(frame_dir: str, frames: list, pano_dir: str):
    """
    Create panoramas with exactly 6 frames each (2 rows × 3 columns)
    Returns list of panorama paths and frame mapping info
    """
    os.makedirs(pano_dir, exist_ok=True)
    pano_paths = []
    pano_info = []

    # Process frames in batches of 6
    for batch_start in range(0, len(frames), 6):
        batch_frames = frames[batch_start:batch_start + 6]

        # Load images
        imgs = []
        for frame_file in batch_frames:
            img_path = os.path.join(frame_dir, frame_file)
            if os.path.exists(img_path):
                imgs.append(Image.open(img_path))

        if not imgs:
            continue

        # Pad with blank images if less than 6 frames
        while len(imgs) < 6:
            # Create blank image with same size as first image
            blank = Image.new("RGB", imgs[0].size, (0, 0, 0))
            imgs.append(blank)

        # Get max dimensions
        max_w = max(img.width for img in imgs)
        max_h = max(img.height for img in imgs)

        # Create panorama canvas (2 rows × 3 columns)
        pano_width = 3 * max_w
        pano_height = 2 * max_h
        pano = Image.new("RGB", (pano_width, pano_height))

        # Frame positions mapping (left to right, top to bottom)
        positions = [
            (0, 0),     # row 1, col 1 - second 0
            (1, 0),     # row 1, col 2 - second 1
            (2, 0),     # row 1, col 3 - second 2
            (0, 1),     # row 2, col 1 - second 3
            (1, 1),     # row 2, col 2 - second 4
            (2, 1)      # row 2, col 3 - second 5
        ]

        # Paste images into grid
        frame_mapping = []
        for idx, (img, (col, row)) in enumerate(zip(imgs, positions)):
            x = col * max_w
            y = row * max_h
            pano.paste(img, (x, y))

            # Calculate actual second in video
            actual_second = batch_start + idx
            frame_mapping.append({
                "grid_position": f"row {row + 1}, column {col + 1}",
                "video_second": actual_second,
                "row": row + 1,
                "column": col + 1
            })

        # Save panorama
        pano_file = f"pano_{batch_start//6:03d}.jpg"
        pano_path = os.path.join(pano_dir, pano_file)
        pano.save(pano_path)

        pano_paths.append(pano_path)
        pano_info.append({
            "panorama_file": pano_file,
            "frame_range": f"{batch_start}-{batch_start + 5} seconds",
            "frame_mapping": frame_mapping
        })

    return pano_paths, pano_info

async def detect_object_in_panorama(pano_path: str, query: str):
    """Send panorama to DeepSeek VL2 for object detection"""

    # Create detection prompt
    prompt = f"""
You are analyzing a panoramic image made of 6 video frames arranged in a 2×3 grid (2 rows, 3 columns).
The frames are arranged left to right, top to bottom.

Grid layout:
[Frame 1] [Frame 2] [Frame 3]
[Frame 4] [Frame 5] [Frame 6]

TASK: Detect if there is "{query}" in any of these frames.

For each detection, specify:
1. Which grid position (row and column number)
2. Confidence level (high/medium/low)
3. Brief description of what you see

Respond in this exact format:
DETECTION_RESULTS:
Position: row X, column Y
Confidence: [high/medium/low]
Description: [what you see]

If multiple detections, list each separately.
If no detection, respond: NO_DETECTION
"""

    try:
        for attempt in range(3):
            try:
                with open(pano_path, "rb") as f:
                    resp = await client.post(
                        DEEPSEEK_API_URL,
                        files={"file": (Path(pano_path).name, f, "image/jpeg")},
                        data={"prompt": prompt, "analysis_type": "object_detection"}
                    )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
    except Exception as e:
        logger.error(f"Error detecting object in panorama: {e}")
        return f"ERROR: {str(e)}"

@app.post("/detect_object/")
async def detect_object_in_video(
    video_url: str = Body(..., embed=True),
    query: str = Body(..., embed=True)
):
    """Main endpoint for object detection in video"""

    logger.info(f"Starting object detection for query: '{query}' in video: {video_url}")

    # Validate video URL format
    parsed_url = urlparse(video_url)
    path_without_query = parsed_url.path
    ext = Path(path_without_query).suffix.lower() if path_without_query else ""

    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(400, f"Unsupported video format: '{ext}'. Supported: .mp4, .avi, .mov")

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
             tempfile.TemporaryDirectory() as frame_dir, \
             tempfile.TemporaryDirectory() as pano_dir:

            # Download video
            logger.info("Downloading video...")
            r = requests.get(video_url, stream=True)
            if r.status_code != 200:
                raise HTTPException(400, "Failed to download video")

            for chunk in r.iter_content(8192):
                tmp_vid.write(chunk)
            tmp_vid.flush()

            # Extract frames at 1 FPS
            logger.info("Extracting frames...")
            frames, duration = extract_1fps(tmp_vid.name, frame_dir)

            # Create panoramas
            logger.info("Creating panoramas...")
            pano_paths, pano_infos = make_panoramas_6_frames(frame_dir, frames, pano_dir)

            # Detect objects in each panorama
            logger.info("Analyzing panoramas for object detection...")
            panorama_responses = []

            for i, (pano_path, pano_info) in enumerate(zip(pano_paths, pano_infos)):
                response = await detect_object_in_panorama(pano_path, query)
                panorama_responses.append({
                    "panorama_index": i,
                    "frame_range": pano_info['frame_range'],
                    "frame_mapping": pano_info['frame_mapping'],
                    "deepseek_response": response
                })

            # Clean up temp video file
            try:
                os.remove(tmp_vid.name)
            except OSError:
                logger.warning(f"Couldn't delete temp file {tmp_vid.name}")

            # Return structured response
            return {
                "query": query,
                "video_duration": round(duration, 2),
                "total_frames_extracted": len(frames),
                "total_panoramas": len(pano_paths),
                "panorama_responses": panorama_responses
            }

    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        raise HTTPException(500, f"Object detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Check Ollama connectivity
    try:
        ollama_resp = await client.get(f"{OLLAMA_CHAT_URL.replace('/api/chat', '')}/api/tags", timeout=5.0)
        health_status["services"]["ollama"] = "connected" if ollama_resp.status_code == 200 else f"error: HTTP {ollama_resp.status_code}"
    except Exception as e:
        health_status["services"]["ollama"] = f"error: {str(e)}"

    # Check DeepSeek connectivity
    try:
        deepseek_resp = await client.get(DEEPSEEK_API_URL.replace('/analyze/', '/health'), timeout=5.0)
        health_status["services"]["deepseek"] = "connected" if deepseek_resp.status_code == 200 else f"error: HTTP {deepseek_resp.status_code}"
    except Exception as e:
        health_status["services"]["deepseek"] = f"error: {str(e)}"

    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
