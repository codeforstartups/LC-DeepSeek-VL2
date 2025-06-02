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
# 7. Helper: Extract frames at 1 FPS from video
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames_1fps(video_path: str, temp_dir: str) -> List[str]:
    """
    Extract frames at 1 FPS from video, similar to video_api.py approach.
    Returns list of extracted frame paths.
    """
    frames_dir = os.path.join(temp_dir, "frames_1fps")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file for frame extraction.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Extracting frames at 1 FPS from video - Duration: {duration:.2f}s, Expected frames: {int(duration)}")

    frame_paths = []
    idx = 0

    while True:
        # Set position to extract frame at each second
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frames_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        if idx % 10 == 0:  # Log every 10 seconds
            logger.info(f"Extracted frame at {idx}s")

        idx += 1

    cap.release()
    logger.info(f"Frame extraction complete - Extracted {len(frame_paths)} frames at 1 FPS")
    return frame_paths

# ──────────────────────────────────────────────────────────────────────────────
# 8. Helper: Process video frames and find matches
# ──────────────────────────────────────────────────────────────────────────────

def match_faces_in_video(local_video_path: str, ref_embeds: Dict[str, np.ndarray]) -> List[MatchResult]:
    """
    Extract frames at 1 FPS first, then process each extracted frame for face matching.
    Much more efficient than processing every frame in the video.
    """
    logger.info(f"Starting face matching in video: {local_video_path}")
    logger.info(f"Will compare against {len(ref_embeds)} reference embeddings")

    # Create temporary directory for extracted frames
    temp_frames_dir = tempfile.mkdtemp()

    try:
        # Step 1: Extract frames at 1 FPS
        frame_paths = extract_frames_1fps(local_video_path, temp_frames_dir)

        if not frame_paths:
            logger.warning("No frames extracted from video")
            return []

        # Step 2: Process each extracted frame
        matches: List[MatchResult] = []
        last_log_time = time.time()

        for frame_idx, frame_path in enumerate(frame_paths):
            # Log progress every 10 frames
            current_time = time.time()
            if current_time - last_log_time >= 10:
                progress = (frame_idx / len(frame_paths) * 100)
                logger.info(f"Processing frame {frame_idx+1}/{len(frame_paths)} ({progress:.1f}%)")
                last_log_time = current_time

            # Calculate timestamp (frame_idx represents seconds since we extracted at 1 FPS)
            timestamp_ms = frame_idx * 1000

            # Load and process frame
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Could not load frame: {frame_path}")
                    continue

                # Compute embedding for the first detected face (if any)
                results = DeepFace.represent(
                    img_path=frame,
                    model_name="ArcFace",
                    enforce_detection=False,
                    detector_backend="opencv"
                )

                if not results or len(results) == 0:
                    logger.debug(f"No face detected in frame {frame_idx} ({frame_path})")
                    continue

                frame_emb = np.array(results[0]["embedding"])
                logger.debug(f"Frame {frame_idx}: Found face, computing matches...")

                # Compare to each reference embedding
                for key, ref_vec in ref_embeds.items():
                    # Cosine similarity: (A·B)/(||A||·||B||); distance = 1 - similarity
                    sim = np.dot(frame_emb, ref_vec) / (np.linalg.norm(frame_emb) * np.linalg.norm(ref_vec) + 1e-10)
                    distance = 1.0 - sim
                    if distance <= 0.4:  # threshold for ArcFace + cosine
                        match = MatchResult(photo_key=key, frame_index=frame_idx, timestamp_ms=timestamp_ms)
                        matches.append(match)
                        logger.info(f"MATCH FOUND! Frame {frame_idx} matches {key} (distance: {distance:.3f}, timestamp: {timestamp_ms}ms)")
                        # Stop checking other references once matched
                        break

            except Exception as e:
                logger.warning(f"Failed to process frame {frame_idx} ({frame_path}): {e}")
                continue

        logger.info(f"Video processing complete. Processed {len(frame_paths)} frames at 1 FPS")
        logger.info(f"Total matches found: {len(matches)}")
        return matches

    finally:
        # Clean up extracted frames
        shutil.rmtree(temp_frames_dir, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────────────
# 9. FastAPI endpoint
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
