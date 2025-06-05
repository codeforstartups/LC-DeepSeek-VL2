import os
import cv2
import httpx
import asyncio
import tempfile
import logging
import requests
import math
import re
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

def parse_detection_response(response_text):
    """Parse the structured response from DeepSeek"""

    # Initialize result
    result = {
        "answer": None,
        "description": "",
        "confidence": None
    }

    # Split by lines and process each
    lines = response_text.strip().split('\n')

    for line in lines:
        line = line.strip()

        # Extract Answer (YES/NO)
        if line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            result["answer"] = answer.upper() == "YES"

        # Extract Description
        elif line.startswith("Description:"):
            result["description"] = line.replace("Description:", "").strip()

        # Extract Confidence (try to get just the number)
        elif line.startswith("Confidence:"):
            conf_text = line.replace("Confidence:", "").strip()
            # Try to extract number (1-10)
            numbers = re.findall(r'\d+', conf_text)
            if numbers:
                result["confidence"] = int(numbers[0])

    return result

async def detect_object_in_frame(frame_path: str, query: str, frame_second: int):
    """Send frame to DeepSeek VL2 for object detection"""

    # Create detection prompt
    prompt_str = f"""Please analyze the image and answer the following questions:
    1. Is there a {query} in the image?
    2. If yes, describe its appearance and location in the image in detail.
    3. If no, describe what you see in the image instead.
    4. On a scale of 1-10, how confident are you in your answer?

    Please structure your response as follows:
    Answer: [YES/NO]
    Description: [Your detailed description]
    Confidence: [1-10]"""

    try:
        for attempt in range(3):
            try:
                with open(frame_path, "rb") as f:
                    resp = await client.post(
                        DEEPSEEK_API_URL,
                        files={"file": (Path(frame_path).name, f, "image/jpeg")},
                        data={"prompt": prompt_str, "analysis_type": "object_detection"}
                    )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except TimeoutException:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
    except Exception as e:
        logger.error(f"Error detecting object in frame: {e}")
        return f"ERROR: {str(e)}"

@app.post("/detect_object/")
async def detect_object_in_video(
    video_url: str = Body(..., embed=True),
    query: str = Body(..., embed=True)
):
    """Main endpoint for object detection in video frames"""

    logger.info(f"Starting object detection for query: '{query}' in video: {video_url}")

    # Validate video URL format
    parsed_url = urlparse(video_url)
    path_without_query = parsed_url.path
    ext = Path(path_without_query).suffix.lower() if path_without_query else ""

    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(400, f"Unsupported video format: '{ext}'. Supported: .mp4, .avi, .mov")

    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
             tempfile.TemporaryDirectory() as frame_dir:

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

            # Analyze each frame
            logger.info("Analyzing frames for object detection...")
            frame_responses = []
            positive_detections = []  # Only YES answers

            for i, frame_file in enumerate(frames):
                frame_path = os.path.join(frame_dir, frame_file)
                response = await detect_object_in_frame(frame_path, query, i)

                # Parse the response
                parsed_result = parse_detection_response(response)

                frame_data = {
                    "second": i,
                    "frame_file": frame_file,
                    "deepseek_response": response,
                    "parsed_result": {
                        "found": parsed_result["answer"],
                        "description": parsed_result["description"],
                        "confidence": parsed_result["confidence"]
                    }
                }

                frame_responses.append(frame_data)

                # Add to positive detections if YES answer
                if parsed_result["answer"]:
                    positive_detections.append({
                        "second": i,
                        "description": parsed_result["description"],
                        "confidence": parsed_result["confidence"]
                    })

            # Clean up temp video file
            try:
                os.remove(tmp_vid.name)
            except OSError:
                logger.warning(f"Couldn't delete temp file {tmp_vid.name}")

            # Calculate summary statistics
            total_frames = len(frames)
            detection_count = len(positive_detections)
            detection_rate = (detection_count / total_frames) * 100 if total_frames > 0 else 0

            # Return structured response
            return {
                "query": query,
                "video_duration": round(duration, 2),
                "total_frames_analyzed": total_frames,
                "detection_summary": {
                    "total_detections": detection_count,
                    "detection_rate": f"{detection_rate:.1f}%",
                    "found_at_seconds": [det["second"] for det in positive_detections]
                },
                "positive_detections": positive_detections,  # Only YES answers
                "frame_responses": frame_responses  # All frames with raw + parsed data
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
