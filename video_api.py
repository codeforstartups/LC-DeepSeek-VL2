import os
import cv2
import httpx
import asyncio
import tempfile
import logging
import requests
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from httpx import HTTPStatusError, TimeoutException

app = FastAPI()
logger = logging.getLogger("analyze_video")
logging.basicConfig(level=logging.INFO)

DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://localhost:8000/infer/")
client = httpx.AsyncClient(timeout=15.0)

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

def make_panoramas(frame_dir: str, frames: list, N: int, pano_dir: str):
    os.makedirs(pano_dir, exist_ok=True)
    panos = []
    for i in range(0, len(frames), N):
        group = frames[i : i + N]
        imgs = [Image.open(f"{frame_dir}/{f}") for f in group]
        widths, heights = zip(*(im.size for im in imgs))
        pano = Image.new("RGB", (sum(widths), max(heights)))
        x = 0
        for im in imgs:
            pano.paste(im, (x, 0))
            x += im.width
        out = f"{pano_dir}/pano_{i//N:03d}.jpg"
        pano.save(out)
        panos.append(out)
    return panos

async def describe_image(path: str, prompt: str):
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
                await asyncio.sleep(2**attempt)
                continue
            raise
        except HTTPStatusError:
            # don’t retry 4xx
            raise

@app.post("/analyze_video/")
async def analyze_video(
    video_url: str = Body(..., embed=True),
    prompt: str = (
        "You're looking at 10 consecutive seconds stitched side-by-side. "
        "Tell the story of what unfolds from left to right—who appears, what actions happen, "
        "and any dynamic changes."
    ),
    N: int = 10,
):
    logger.info(f"Downloading video from URL: {video_url}")
    ext = Path(video_url).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(400, "Unsupported video format")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # download
        r = requests.get(video_url, stream=True)
        if r.status_code != 200:
            raise HTTPException(400, "Failed to download video")
        for chunk in r.iter_content(8192):
            tmp_vid.write(chunk)
        tmp_vid.flush()

        # extract + stitch
        frames = extract_1fps(tmp_vid.name, frame_dir)
        panos  = make_panoramas(frame_dir, frames, N, pano_dir)

        # describe one by one
        descriptions = []
        for p in panos:
            name = Path(p).name
            try:
                desc = await describe_image(p, prompt)
                descriptions.append({"pano": name, "description": desc})
            except TimeoutException:
                descriptions.append({"pano": name, "error": "DeepSeek request timed out"})
            except HTTPStatusError as e:
                descriptions.append({"pano": name, "error": f"DeepSeek HTTP {e.response.status_code}"})
            except Exception as e:
                descriptions.append({"pano": name, "error": str(e)})

    try:
        os.remove(tmp_vid.name)
    except OSError:
        logger.warning(f"Couldn't delete temp file {tmp_vid.name}")

    return {"panoramas_analyzed": len(panos), "descriptions": descriptions}
