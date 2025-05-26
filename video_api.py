import os
import cv2
import httpx
import asyncio
import tempfile
import logging
import requests
import boto3
import math
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from httpx import HTTPStatusError, TimeoutException
from urllib.parse import urlparse

app = FastAPI()
logger = logging.getLogger("analyze_video")
logging.basicConfig(level=logging.INFO)

# DeepSeek-VL2 API URL (unchanged)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://localhost:8000/infer/")
# Ollama chat API for summarization
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")

# Log configuration on startup
logger.info(f"DeepSeek API URL: {DEEPSEEK_API_URL}")
logger.info(f"Ollama Chat URL: {OLLAMA_CHAT_URL}")
logger.info(f"S3 Bucket: {os.getenv('AWS_S3_BUCKET', 'Not set')}")

client = httpx.AsyncClient(timeout=None)

# S3 bucket name (unchanged)
S3_BUCKET = os.getenv("AWS_S3_BUCKET")
s3 = boto3.client("s3")

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

def extract_1fps(video_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * idx)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{out_dir}/frame_{idx:03d}.jpg", frame)
        idx += 1
    cap.release()
    return sorted(os.listdir(out_dir))

def make_panoramas(
    frame_dir: str,
    frames: list,
    N: int,
    pano_dir: str,
    cols: int = 5
):
    """
    Stitch up to N frames into a grid with 'cols' columns (and as many rows as needed).
    Returns list of saved panorama file paths.
    """
    os.makedirs(pano_dir, exist_ok=True)
    pano_paths = []

    for batch_start in range(0, len(frames), N):
        group = frames[batch_start : batch_start + N]
        imgs  = [Image.open(os.path.join(frame_dir, f)) for f in group]

        # Determine cell size
        max_w = max(im.width for im in imgs)
        max_h = max(im.height for im in imgs)

        # Calculate number of rows needed
        rows = math.ceil(len(imgs) / cols)

        # Create blank canvas
        pano = Image.new("RGB", (cols * max_w, rows * max_h))

        # Paste each image into its grid cell
        for idx, im in enumerate(imgs):
            row = idx // cols
            col = idx % cols
            x = col * max_w
            y = row * max_h
            pano.paste(im, (x, y))

        out_path = os.path.join(pano_dir, f"pano_{batch_start//N:03d}.jpg")
        pano.save(out_path)
        pano_paths.append(out_path)

    return pano_paths

async def describe_image(path: str, prompt: str):
    # unchanged
    for attempt in range(3):
        try:
            with open(path, "rb") as f:
                resp = await client.post(
                    DEEPSEEK_API_URL,
                    files={"file": (Path(path).name, f, "image/jpeg")},
                    data={"prompt": prompt},
                )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except TimeoutException:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except HTTPStatusError:
            raise

DEFAULT_PROMPT = (
    "You're looking at N consecutive seconds arranged in a grid of 'cols' columns. "
    "Walk me through each row, left to right, describing:\n"
    "  • Who or what enters or exits each cell\n"
    "  • Any actions or interactions you see\n"
    "  • Changes in the scene's context or lighting\n"
    "Keep it concise but chronological."
)

@app.post("/analyze_video/")
async def analyze_video(
    video_url: str = Body(..., embed=True),
    prompt: str = DEFAULT_PROMPT,
    N: int = 30,
    cols: int = 5,
):
    if S3_BUCKET is None:
        raise HTTPException(500, "AWS_S3_BUCKET environment variable not set")

    # (1) Download & extract panoramas (unchanged) …
    logger.info(f"Downloading video from URL: {video_url}")
    parsed_url = urlparse(video_url)
    path_without_query = parsed_url.path
    ext = Path(path_without_query).suffix.lower() if     path_without_query else ""

    logger.info(f"Detected file extension: '{ext}' from path: {path_without_query}")

    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(400, f"Unsupported video format. Detected extension: '{ext}'. Supported formats: .mp4, .avi, .mov")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # download video …
        r = requests.get(video_url, stream=True)
        if r.status_code != 200:
            raise HTTPException(400, "Failed to download video")
        for chunk in r.iter_content(8192):
            tmp_vid.write(chunk)
        tmp_vid.flush()

        # frame extraction & panorama creation …
        frames = extract_1fps(tmp_vid.name, frame_dir)
        panos  = make_panoramas(frame_dir, frames, N, pano_dir, cols)

        # upload panoramas to S3 …
        s3_urls = []
        for pano_path in panos:
            key = f"panoramas/{Path(pano_path).name}"
            s3.upload_file(pano_path, S3_BUCKET, key)
            s3_urls.append(f"https://{S3_BUCKET}.s3.amazonaws.com/{key}")

        # describe each panorama …
        descriptions = []
        for pano_path, s3_url in zip(panos, s3_urls):
            name = Path(pano_path).name
            try:
                desc = await describe_image(pano_path, prompt)
                descriptions.append({
                    "pano": name,
                    "s3_url": s3_url,
                    "description": desc
                })
            except TimeoutException:
                descriptions.append({
                    "pano": name,
                    "s3_url": s3_url,
                    "error": "DeepSeek request timed out"
                })
            except HTTPStatusError as e:
                descriptions.append({
                    "pano": name,
                    "s3_url": s3_url,
                    "error": f"DeepSeek HTTP {e.response.status_code}"
                })
            except Exception as e:
                descriptions.append({
                    "pano": name,
                    "s3_url": s3_url,
                    "error": str(e)
                })

    # (2) Summarize all descriptions via DeepSeek-R1
    all_text = "\n\n".join(item.get("description", "") for item in descriptions if item.get("description"))

    # Check if we have any valid descriptions to summarize
    if not all_text.strip():
        logger.warning("No valid descriptions found for summarization")
        summary = "No valid descriptions were generated from the video analysis."
    else:
        chat_payload = {
            "model": "deepseek-r1:1.5b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that reads multiple scene descriptions "
                        "and produces a concise 2–3 sentence summary."
                    )
                },
                {"role": "user", "content": all_text}
            ],
            "stream": False
        }

        try:
            logger.info(f"Sending summarization request to {OLLAMA_CHAT_URL}")
            logger.debug(f"Payload: {chat_payload}")

            summary_resp = await client.post(OLLAMA_CHAT_URL, json=chat_payload)
            summary_resp.raise_for_status()

            response_data = summary_resp.json()
            logger.debug(f"Ollama response: {response_data}")

            summary = response_data.get("message", {}).get("content", "")
            if not summary:
                logger.warning("Empty summary received from Ollama")
                summary = "Summary generation completed but returned empty content."

        except httpx.ConnectError as e:
            logger.error(f"Connection failed to Ollama at {OLLAMA_CHAT_URL}: {e}")
            summary = f"Summary generation failed: Unable to connect to Ollama service at {OLLAMA_CHAT_URL}"
        except httpx.TimeoutException as e:
            logger.error(f"Timeout while connecting to Ollama: {e}")
            summary = "Summary generation failed: Request to Ollama service timed out"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}")
            summary = f"Summary generation failed: Ollama service returned HTTP {e.response.status_code}"
        except KeyError as e:
            logger.error(f"Unexpected response format from Ollama: {e}")
            summary = "Summary generation failed: Unexpected response format from Ollama service"
        except Exception as e:
            logger.error(f"Unexpected error during summary generation: {type(e).__name__}: {e}")
            summary = f"Summary generation failed: {type(e).__name__}: {str(e)}"

    # cleanup
    try:
        os.remove(tmp_vid.name)
    except OSError:
        logger.warning(f"Couldn't delete temp file {tmp_vid.name}")

    # (3) Return everything, including the new summary
    return {
        "panoramas_analyzed": len(panos),
        "s3_urls": s3_urls,
        "descriptions": descriptions,
        "summary": summary
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service connectivity"""
    health_status = {
        "status": "healthy",
        "services": {}
    }

    # Check Ollama connectivity
    try:
        ollama_resp = await client.get(f"{OLLAMA_CHAT_URL.replace('/api/chat', '')}/api/tags", timeout=5.0)
        if ollama_resp.status_code == 200:
            health_status["services"]["ollama"] = "connected"
        else:
            health_status["services"]["ollama"] = f"error: HTTP {ollama_resp.status_code}"
    except Exception as e:
        health_status["services"]["ollama"] = f"error: {str(e)}"

    # Check DeepSeek connectivity
    try:
        deepseek_resp = await client.get(DEEPSEEK_API_URL.replace('/infer/', '/health'), timeout=5.0)
        if deepseek_resp.status_code == 200:
            health_status["services"]["deepseek"] = "connected"
        else:
            health_status["services"]["deepseek"] = f"error: HTTP {deepseek_resp.status_code}"
    except Exception as e:
        health_status["services"]["deepseek"] = f"error: {str(e)}"

    # Check S3 configuration
    if S3_BUCKET:
        health_status["services"]["s3"] = f"configured: {S3_BUCKET}"
    else:
        health_status["services"]["s3"] = "error: AWS_S3_BUCKET not set"

    return health_status
