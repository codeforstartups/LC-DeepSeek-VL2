import os
import tempfile
import shutil
import time
import boto3
import cv2
import numpy as np
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────
# 1. FastAPI app initialization
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Pydantic models for request and response
# ──────────────────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    video_url: str         # Publicly accessible HTTP(S) URL for the video
    photos_bucket: str     # S3 bucket name containing reference images
    photos_prefix: str     # S3 key prefix (i.e., folder) under which all reference images reside

class MatchResult(BaseModel):
    photo_key: str         # S3 key of the matched reference image
    frame_index: int       # Frame index in the video where match occurred
    timestamp_ms: int      # Timestamp (in milliseconds) of that frame

class MatchResponse(BaseModel):
    matches: List[MatchResult]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper: Download video from URL to a local temp file
# ──────────────────────────────────────────────────────────────────────────────

def download_video(video_url: str) -> str:
    """
    Download the video from the given URL into a temporary local file.
    Returns the local filepath.
    """
    # Create a temp file
    temp_dir = tempfile.mkdtemp()
    local_video_path = os.path.join(temp_dir, "input_video.mp4")
    try:
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(local_video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to download video: {e}")
    return local_video_path

# ──────────────────────────────────────────────────────────────────────────────
# 4. Helper: List all object keys under a given S3 prefix
# ──────────────────────────────────────────────────────────────────────────────

def list_s3_keys(bucket_name: str, prefix: str) -> List[str]:
    """
    Return a list of all S3 keys under the given bucket and prefix.
    Skips any 'keys' ending with '/' (i.e., folder markers).
    """
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    keys: List[str] = []
    for page in page_iterator:
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)
    return keys

# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper: Download a single S3 object to a local path
# ──────────────────────────────────────────────────────────────────────────────

def download_s3_object(bucket: str, key: str, local_path: str):
    """
    Download the S3 object at 'key' in 'bucket' into 'local_path'.
    Ensures the directory exists.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client("s3").download_file(Bucket=bucket, Key=key, Filename=local_path)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Helper: Compute embeddings for all reference images
# ──────────────────────────────────────────────────────────────────────────────

def compute_reference_embeddings(bucket: str, keys: List[str], local_dir: str) -> Dict[str, np.ndarray]:
    """
    Given a list of S3 keys, download each image into local_dir,
    compute its DeepFace.embedding (ArcFace+cosine), and return a dict mapping:
      key -> embedding (numpy array)
    """
    embeddings: Dict[str, np.ndarray] = {}
    for key in keys:
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        try:
            download_s3_object(bucket, key, local_path)
        except Exception as e:
            # Skip if download fails
            continue

        try:
            # Compute embedding with ArcFace (512-D) and cosine metric
            # enforce_detection=True ensures we only keep images with a visible face
            result = DeepFace.represent(
                img_path=local_path,
                model_name="ArcFace",
                enforce_detection=True,
                detector_backend="opencv"
            )
            if result and len(result) > 0:
                emb_vector = np.array(result[0]["embedding"])
                embeddings[key] = emb_vector
        except Exception:
            # Skip images where face detection/embedding fails
            continue
    return embeddings

# ──────────────────────────────────────────────────────────────────────────────
# 7. Helper: Process video frames and find matches
# ──────────────────────────────────────────────────────────────────────────────

def match_faces_in_video(local_video_path: str, ref_embeds: Dict[str, np.ndarray]) -> List[MatchResult]:
    """
    Open the local video file, iterate frame by frame, compute each frame's embedding,
    and compare to each reference embedding. Whenever cosine distance ≤ 0.4,
    record a MatchResult (photo_key, frame_index, timestamp_ms).
    """
    matches: List[MatchResult] = []
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration_ms = 1000.0 / fps

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute timestamp for this frame (in milliseconds)
        pos_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if pos_msec <= 0:
            pos_msec = int(frame_index * frame_duration_ms)

        # Compute embedding for the first detected face (if any)
        try:
            results = DeepFace.represent(
                img_path=frame,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="opencv"
            )
        except Exception:
            frame_index += 1
            continue

        if not results or len(results) == 0:
            frame_index += 1
            continue

        frame_emb = np.array(results[0]["embedding"])

        # Compare to each reference embedding
        for key, ref_vec in ref_embeds.items():
            # Cosine similarity: (A·B)/(||A||·||B||); distance = 1 - similarity
            sim = np.dot(frame_emb, ref_vec) / (np.linalg.norm(frame_emb) * np.linalg.norm(ref_vec) + 1e-10)
            distance = 1.0 - sim
            if distance <= 0.4:  # threshold for ArcFace + cosine
                matches.append(
                    MatchResult(photo_key=key, frame_index=frame_index, timestamp_ms=pos_msec)
                )
                # Stop checking other references once matched
                break

        frame_index += 1

    cap.release()
    return matches

# ──────────────────────────────────────────────────────────────────────────────
# 8. FastAPI endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/match_faces", response_model=MatchResponse)
def match_faces(request: MatchRequest):
    """
    Accepts:
      - video_url: HTTP(S) URL to a video file
      - photos_bucket: S3 bucket where reference images are stored
      - photos_prefix: S3 key prefix (folder) under which reference images reside

    Returns:
      - A JSON array of matches, where each match has:
          photo_key, frame_index, timestamp_ms
    """
    # 1. Download the video to a temporary file
    try:
        local_video = download_video(request.video_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Video download failed: {e}")

    # 2. List all reference image keys in the S3 folder
    try:
        photo_keys = list_s3_keys(request.photos_bucket, request.photos_prefix)
    except Exception as e:
        # Clean up video file
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to list S3 keys: {e}")

    if not photo_keys:
        # Clean up video file
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        return MatchResponse(matches=[])

    # 3. Download each image and compute embeddings
    temp_photos_dir = tempfile.mkdtemp()
    try:
        ref_embeddings = compute_reference_embeddings(
            request.photos_bucket, photo_keys, temp_photos_dir
        )
    except Exception as e:
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to compute embeddings: {e}")

    if not ref_embeddings:
        # No valid embeddings; cleanup and return empty
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        return MatchResponse(matches=[])

    # 4. Process the video and find matches
    try:
        results = match_faces_in_video(local_video, ref_embeddings)
    except Exception as e:
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed during face matching: {e}")

    # 5. Cleanup temporary files
    shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
    shutil.rmtree(temp_photos_dir, ignore_errors=True)

    # 6. Return results
    return MatchResponse(matches=results)
