import os
import httpx
import asyncio
import tempfile
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI()

# DeepSeek-VL2 API URL (env var or default)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "http://localhost:8000/infer/")

# Helper: extract 1 FPS frames
def extract_1fps(video_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    import cv2
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * idx)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:03d}.jpg"), frame)
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
    "You’re looking at 10 consecutive seconds stitched side-by-side. "
    "Tell the story of what unfolds from left to right—who appears, what actions happen, "
    "and any dynamic changes."
)

):
    # 1) Validate extension
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    # 2) Create temps for vid, frames, panos
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_vid, \
         tempfile.TemporaryDirectory() as frame_dir, \
         tempfile.TemporaryDirectory() as pano_dir:

        # save upload
        tmp_vid.write(await video.read())
        tmp_vid.flush()

        # 3) extract & stitch
        frame_files = extract_1fps(tmp_vid.name, frame_dir)
        pano_paths  = make_panoramas(frame_dir, frame_files, pano_dir=pano_dir)

        # 4) describe all panoramas in parallel
        async with httpx.AsyncClient(timeout=15.0) as client:
            tasks = [describe_image(client, p, prompt) for p in pano_paths]
            captions = await asyncio.gather(*tasks, return_exceptions=True)

        # prepare output
        descriptions = []
        for path, cap in zip(pano_paths, captions):
            if isinstance(cap, Exception):
                descriptions.append({
                    "pano": Path(path).name,
                    "error": str(cap)
                })
            else:
                descriptions.append({
                    "pano": Path(path).name,
                    "description": cap
                })

    # 5) cleanup temp video
    try:
        os.remove(tmp_vid.name)
    except OSError:
        pass

    return {
        "panoramas_analyzed": len(pano_paths),
        "descriptions": descriptions
    }
