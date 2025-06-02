import os
import tempfile
import shutil
import time
import boto3
import cv2
import numpy as np
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Configure logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 0. Enable TensorFlow GPU Memory Growth (must run before any DeepFace calls)
# ──────────────────────────────────────────────────────────────────────────────
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # (Optional) Restrict TensorFlow to GPU:0 only
    # tf.config.set_visible_devices(gpus[0], "GPU")

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
    logger.info(f"Starting video download from: {video_url}")
    temp_dir = tempfile.mkdtemp()
    local_video_path = os.path.join(temp_dir, "input_video.mp4")
    try:
        logger.info(f"Downloading to temporary path: {local_video_path}")
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            logger.info(f"Video file size: {total_size} bytes")

            downloaded = 0
            with open(local_video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and downloaded % (1024*1024) == 0:  # Log every MB
                            logger.info(f"Downloaded: {downloaded}/{total_size} bytes ({100*downloaded/total_size:.1f}%)")

        logger.info(f"Video download completed successfully: {local_video_path}")
    except Exception as e:
        logger.error(f"Failed to download video: {e}")
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
    logger.info(f"Listing S3 keys in bucket: {bucket_name}, prefix: {prefix}")
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

    logger.info(f"Found {len(keys)} reference images: {keys}")
    return keys

# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper: Download a single S3 object to a local path
# ──────────────────────────────────────────────────────────────────────────────

def download_s3_object(bucket: str, key: str, local_path: str):
    """
    Download the S3 object at 'key' in 'bucket' into 'local_path'.
    Ensures the directory exists.
    """
    logger.debug(f"Downloading S3 object: s3://{bucket}/{key} -> {local_path}")
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
    logger.info(f"Computing embeddings for {len(keys)} reference images")
    embeddings: Dict[str, np.ndarray] = {}

    for i, key in enumerate(keys):
        logger.info(f"Processing reference image {i+1}/{len(keys)}: {key}")
        filename = os.path.basename(key)
        local_path = os.path.join(local_dir, filename)
        try:
            download_s3_object(bucket, key, local_path)
            logger.debug(f"Downloaded: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to download {key}: {e}")
            continue

        try:
            # Compute embedding with ArcFace (512-D) and cosine metric
            # enforce_detection=True ensures we only keep images with a visible face
            logger.debug(f"Computing embedding for: {key}")
            result = DeepFace.represent(
                img_path=local_path,
                model_name="ArcFace",
                enforce_detection=True,
                detector_backend="opencv"
            )
            if result and len(result) > 0:
                emb_vector = np.array(result[0]["embedding"])
                embeddings[key] = emb_vector
                logger.info(f"Successfully computed embedding for: {key} (shape: {emb_vector.shape})")
            else:
                logger.warning(f"No face detected in: {key}")
        except Exception as e:
            # Skip images where face detection/embedding fails
            logger.warning(f"Failed to compute embedding for {key}: {e}")
            continue

    logger.info(f"Successfully computed {len(embeddings)} embeddings out of {len(keys)} images")
    return embeddings

# ──────────────────────────────────────────────────────────────────────────────
# 7. Helper: Process video frames and find matches
# ──────────────────────────────────────────────────────────────────────────────

def match_faces_in_video(local_video_path: str, ref_embeds: Dict[str, np.ndarray]) -> List[MatchResult]:
    """
    Open the local video file, iterate frame by frame, compute each frame's embedding,
    and compare to each reference embedding. Whenever cosine distance ≤ 0.4,
    record a MatchResult (photo_key, frame_index, timestamp_ms).

    OPTIMIZED: Process at 1 FPS instead of full frame rate for efficiency.
    """
    logger.info(f"Starting face matching in video: {local_video_path}")
    logger.info(f"Will compare against {len(ref_embeds)} reference embeddings")

    matches: List[MatchResult] = []
    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        logger.error("Cannot open video file")
        raise RuntimeError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration_ms = 1000.0 / fps

    # Calculate frame skip to achieve ~1 FPS processing
    frame_skip = max(1, int(fps))  # Process every Nth frame to get ~1 FPS

    logger.info(f"Video info - FPS: {fps}, Total frames: {total_frames}, Duration: {total_frames/fps:.2f}s")
    logger.info(f"OPTIMIZATION: Processing every {frame_skip} frames (~1 FPS) instead of all {fps} FPS")
    logger.info(f"Expected processing frames: {total_frames // frame_skip}")

    frame_index = 0
    processed_frames = 0
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every Nth frame to achieve ~1 FPS
        if frame_index % frame_skip == 0:
            processed_frames += 1

            # Log progress every 10 seconds
            current_time = time.time()
            if current_time - last_log_time >= 10:
                progress = (frame_index / total_frames * 100) if total_frames > 0 else 0
                logger.info(f"Processing frame {frame_index}/{total_frames} ({progress:.1f}%) - Processed: {processed_frames}")
                last_log_time = current_time

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
            except Exception as e:
                logger.debug(f"Failed to process frame {frame_index}: {e}")
                frame_index += 1
                continue

            if not results or len(results) == 0:
                logger.debug(f"No face detected in frame {frame_index}")
                frame_index += 1
                continue

            frame_emb = np.array(results[0]["embedding"])
            logger.debug(f"Frame {frame_index}: Found face, computing matches...")

            # Compare to each reference embedding
            for key, ref_vec in ref_embeds.items():
                # Cosine similarity: (A·B)/(||A||·||B||); distance = 1 - similarity
                sim = np.dot(frame_emb, ref_vec) / (np.linalg.norm(frame_emb) * np.linalg.norm(ref_vec) + 1e-10)
                distance = 1.0 - sim
                if distance <= 0.4:  # threshold for ArcFace + cosine
                    match = MatchResult(photo_key=key, frame_index=frame_index, timestamp_ms=pos_msec)
                    matches.append(match)
                    logger.info(f"MATCH FOUND! Frame {frame_index} matches {key} (distance: {distance:.3f}, timestamp: {pos_msec}ms)")
                    # Stop checking other references once matched
                    break

        frame_index += 1

    cap.release()
    logger.info(f"Video processing complete. Processed {processed_frames} frames out of {total_frames} total frames")
    logger.info(f"Total matches found: {len(matches)}")
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
    start_time = time.time()
    logger.info("="*80)
    logger.info("NEW FACE MATCHING REQUEST")
    logger.info(f"Video URL: {request.video_url}")
    logger.info(f"Photos bucket: {request.photos_bucket}")
    logger.info(f"Photos prefix: {request.photos_prefix}")
    logger.info("="*80)

    # 1. Download the video to a temporary file
    try:
        local_video = download_video(request.video_url)
    except Exception as e:
        logger.error(f"Video download failed: {e}")
        raise HTTPException(status_code=400, detail=f"Video download failed: {e}")

    # 2. List all reference image keys in the S3 folder
    try:
        photo_keys = list_s3_keys(request.photos_bucket, request.photos_prefix)
    except Exception as e:
        logger.error(f"Failed to list S3 keys: {e}")
        # Clean up video file
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to list S3 keys: {e}")

    if not photo_keys:
        logger.warning("No reference images found in S3")
        # Clean up video file
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        return MatchResponse(matches=[])

    # 3. Download each image and compute embeddings
    temp_photos_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory for photos: {temp_photos_dir}")

    try:
        ref_embeddings = compute_reference_embeddings(
            request.photos_bucket, photo_keys, temp_photos_dir
        )
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to compute embeddings: {e}")

    if not ref_embeddings:
        logger.warning("No valid embeddings computed from reference images")
        # No valid embeddings; cleanup and return empty
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        return MatchResponse(matches=[])

    # 4. Process the video and find matches
    try:
        results = match_faces_in_video(local_video, ref_embeddings)
    except Exception as e:
        logger.error(f"Failed during face matching: {e}")
        shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
        shutil.rmtree(temp_photos_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed during face matching: {e}")

    # 5. Cleanup temporary files
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(os.path.dirname(local_video), ignore_errors=True)
    shutil.rmtree(temp_photos_dir, ignore_errors=True)

    # 6. Return results
    elapsed_time = time.time() - start_time
    logger.info("="*80)
    logger.info(f"FACE MATCHING COMPLETE - Total time: {elapsed_time:.2f}s")
    logger.info(f"Found {len(results)} matches")
    for i, match in enumerate(results):
        logger.info(f"  Match {i+1}: {match.photo_key} at frame {match.frame_index} ({match.timestamp_ms}ms)")
    logger.info("="*80)

    return MatchResponse(matches=results)
