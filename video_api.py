import os
import cv2
import httpx
import asyncio
import tempfile
import logging
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from httpx import HTTPStatusError, TimeoutException
import requests

app = FastAPI()

# — Logging setup —
logger = logging.getLogger("analyze_video")
logging.basicConfig(level=logging.INFO)

# DeepSeek-VL2 API URL (env var or default)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://localhost:8000/infer/")

# Reusable HTTP client
client = httpx.AsyncClient(timeout=15.0)

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()

# Helper: extract 1 FPS frames
def extract_1fps(video_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return sorted(os.listdir(out_dir))

# Helper: stitch frames into panoramas of N
def make_panoramas(frame_dir: str, frames: list, N: int = 10, pano_dir: str = None):
    pano_dir = pano_dir or os.path.join(frame_dir, "panos")
    os.makedirs(pano_dir, exist_ok=True)
    pano_paths = []
    for i in range(0, len(frames), N):
        group = frames[i:i+N]
        imgs = [Image.open(os.path.join(frame_dir, f)) for f in group]
        widths, heights = zip(*(im.size for im in imgs))
        pano = Image.new('RGB', (sum(widths), max(heights)))
        x_offset = 0
        for im in imgs:
            pano.paste(im, (x_offset, 0))
            x_offset += im.width
        pano_path = os.path.join(pano_dir, f"pano_{i//N:03d}.jpg")
        pano.save(pano_path)
        pano_paths.append(pano_path)
    return pano_paths

# Helper: describe an image using DeepSeek-VL2 API
async def describe_image(client: httpx.AsyncClient, path: str, prompt: str):
    with open(path, 'rb') as f:
        files = {"file": (Path(path).name, f, "image/jpeg")}
        data = {"prompt": prompt}
        resp = await client.post(DEEPSEEK_API_URL, files=files, data=data)
    resp.raise_for_status()
    return resp.json().get("response", "")

@app.post("/analyze_video/")
async def analyze_video(
    video: UploadFile = File(...),
    prompt: str = (
        "You're looking at 10 consecutive seconds stitched side-by-side. "
        "Tell the story of what unfolds from left to right—who appears, what actions happen, "
        "and any dynamic changes."
    ),
    N: int = 10
):
    logger.info(f"Received video: {video.filename}")
    # 1) Validate extension
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # 2) Create temps for vid + frames + panos
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # save upload
        content = await video.read()
        tmp_vid.write(content)
        tmp_vid.flush()

        # 3) extract & stitch
        frame_files = extract_1fps(tmp_vid.name, frame_dir)
        logger.info(f"Extracted {len(frame_files)} frames at 1 FPS")
        pano_paths = make_panoramas(frame_dir, frame_files, N=N, pano_dir=pano_dir)
        logger.info(f"Created {len(pano_paths)} panoramas with batch size N={N}")

        # 4) describe all panoramas in parallel
        tasks = [describe_image(client, p, prompt) for p in pano_paths]
        captions = await asyncio.gather(*tasks, return_exceptions=True)

        # prepare output with detailed error handling
        descriptions = []
        for path, cap in zip(pano_paths, captions):
            name = Path(path).name
            if isinstance(cap, Exception):
                if isinstance(cap, TimeoutException):
                    err = "DeepSeek request timed out"
                elif isinstance(cap, HTTPStatusError):
                    err = f"DeepSeek HTTP {cap.response.status_code}"
                else:
                    err = str(cap)
                logger.error(f"Error on {name}: {err}")
                descriptions.append({"pano": name, "error": err})
            else:
                descriptions.append({"pano": name, "description": cap})

    # 5) Cleanup temp video
    try:
        os.remove(tmp_vid.name)
    except OSError:
        logger.warning(f"Failed to delete temp file {tmp_vid.name}")

    return {
        "panoramas_analyzed": len(pano_paths),
        "descriptions": descriptions
    }

@app.post("/analyze_static/")
async def analyze_static(
    prompt: str = (
        "You're looking at 10 consecutive seconds stitched side-by-side. "
        "Tell the story of what unfolds from left to right—who appears, what actions happen, "
        "and any dynamic changes."
    ),
    N: int = 10
):
    static_video = "test_1.mp4"
    if not os.path.exists(static_video):
        raise HTTPException(status_code=404, detail=f"Static video '{static_video}' not found.")
    ext = Path(static_video).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # Copy static video to temp file
        with open(static_video, "rb") as src:
            tmp_vid.write(src.read())
        tmp_vid.flush()

        # 3) extract & stitch
        frame_files = extract_1fps(tmp_vid.name, frame_dir)
        logger.info(f"Extracted {len(frame_files)} frames at 1 FPS from static video")
        pano_paths = make_panoramas(frame_dir, frame_files, N=N, pano_dir=pano_dir)
        logger.info(f"Created {len(pano_paths)} panoramas with batch size N={N} from static video")

        # 4) describe all panoramas in parallel
        tasks = [describe_image(client, p, prompt) for p in pano_paths]
        captions = await asyncio.gather(*tasks, return_exceptions=True)

        # prepare output with detailed error handling
        descriptions = []
        for path, cap in zip(pano_paths, captions):
            name = Path(path).name
            if isinstance(cap, Exception):
                if isinstance(cap, TimeoutException):
                    err = "DeepSeek request timed out"
                elif isinstance(cap, HTTPStatusError):
                    err = f"DeepSeek HTTP {cap.response.status_code}"
                else:
                    err = str(cap)
                logger.error(f"Error on {name}: {err}")
                descriptions.append({"pano": name, "error": err})
            else:
                descriptions.append({"pano": name, "description": cap})

    # 5) Cleanup temp video
    try:
        os.remove(tmp_vid.name)
    except OSError:
        logger.warning(f"Failed to delete temp file {tmp_vid.name}")

    return {
        "panoramas_analyzed": len(pano_paths),
        "descriptions": descriptions
    }

@app.post("/analyze_from_url/")
async def analyze_from_url(
    video_url: str = Body(..., embed=True),
    prompt: str = (
        "You're looking at 10 consecutive seconds stitched side-by-side. "
        "Tell the story of what unfolds from left to right—who appears, what actions happen, "
        "and any dynamic changes."
    ),
    N: int = 10
):
    logger.info(f"Received video URL: {video_url}")
    ext = Path(video_url).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # Download video from URL
        r = requests.get(video_url, stream=True)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download video from URL")
        for chunk in r.iter_content(chunk_size=8192):
            tmp_vid.write(chunk)
        tmp_vid.flush()

        # 3) extract & stitch
        frame_files = extract_1fps(tmp_vid.name, frame_dir)
        logger.info(f"Extracted {len(frame_files)} frames at 1 FPS from URL video")
        pano_paths = make_panoramas(frame_dir, frame_files, N=N, pano_dir=pano_dir)
        logger.info(f"Created {len(pano_paths)} panoramas with batch size N={N} from URL video")

        # 4) describe all panoramas in parallel
        tasks = [describe_image(client, p, prompt) for p in pano_paths]
        captions = await asyncio.gather(*tasks, return_exceptions=True)

        # prepare output with detailed error handling
        descriptions = []
        for path, cap in zip(pano_paths, captions):
            name = Path(path).name
            if isinstance(cap, Exception):
                if isinstance(cap, TimeoutException):
                    err = "DeepSeek request timed out"
                elif isinstance(cap, HTTPStatusError):
                    err = f"DeepSeek HTTP {cap.response.status_code}"
                else:
                    err = str(cap)
                logger.error(f"Error on {name}: {err}")
                descriptions.append({"pano": name, "error": err})
            else:
                descriptions.append({"pano": name, "description": cap})

    # 5) Cleanup temp video
    try:
        os.remove(tmp_vid.name)
    except OSError:
        logger.warning(f"Failed to delete temp file {tmp_vid.name}")

    return {
        "panoramas_analyzed": len(pano_paths),
        "descriptions": descriptions
    }
