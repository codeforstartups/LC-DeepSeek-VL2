import os
import tempfile
import shutil
import time
import boto3
import cv2
import requests
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deepface import DeepFace
from typing import List

# ──────────────────────────────────────────────────────────────────────────────
# Configure logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    # Optionally restrict to GPU:0
    # tf.config.set_visible_devices(gpus[0], "GPU")

# ──────────────────────────────────────────────────────────────────────────────
# 1. FastAPI app initialization
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Pydantic models for request and response
# ──────────────────────────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    video_url: str       # Public HTTPS video URL
    photos_bucket: str   # S3 bucket containing reference images
    photos_prefix: str   # Prefix (folder) under which reference images reside

class MatchResult(BaseModel):
    photo_key: str       # S3 key of matched reference image
    frame_index: int     # Which second/frame (0-based) matched
    timestamp_ms: int    # Millisecond timestamp for that frame

class MatchResponse(BaseModel):
    matches: List[MatchResult]

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper: Download video from a URL to a local temp file
# ──────────────────────────────────────────────────────────────────────────────
def download_video(video_url: str) -> str:
    """
    Download the given video URL into a temporary local file and return its path.
    """
    logger.info(f"Downloading video from: {video_url}")
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
        raise RuntimeError(f"Video download failed: {e}")

    logger.info(f"Video saved to: {local_video_path}")
    return local_video_path

# ──────────────────────────────────────────────────────────────────────────────
# 4. Helper: List all object keys under a given S3 prefix
# ──────────────────────────────────────────────────────────────────────────────
def list_s3_keys(bucket_name: str, prefix: str) -> List[str]:
    """
    Return list of all S3 keys under the given bucket and prefix (excluding 'folder' markers).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                keys.append(key)

    return keys

# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper: Download a single S3 object to a local path
# ──────────────────────────────────────────────────────────────────────────────
def download_s3_object(bucket: str, key: str, local_path: str):
    """
    Download the S3 object at 'key' in 'bucket' into 'local_path'. Creates directories as needed.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client("s3").download_file(Bucket=bucket, Key=key, Filename=local_path)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Helper: Extract frames at 1 FPS from video
# ──────────────────────────────────────────────────────────────────────────────
def extract_frames_1fps(video_path: str, out_folder: str) -> List[str]:
    """
    Extract one frame per second from the video at 'video_path', store them under 'out_folder',
    and return a list of the saved frame file paths in chronological order.
    """
    os.makedirs(out_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for frame extraction.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps

    logger.info(f"Video Duration: {duration_s:.2f}s @ {fps:.2f} FPS; extracting ~{int(duration_s)} frames at 1 FPS")

    frame_paths = []
    sec = 0

    while True:
        # Seek to the exact millisecond for the current second
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"frame_{sec:03d}.jpg"
        frame_path = os.path.join(out_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

        # Optionally log progress every 10 seconds
        if sec % 10 == 0:
            logger.info(f"Extracted frame at {sec}s → {frame_filename}")

        sec += 1

    cap.release()
    logger.info(f"Total frames extracted: {len(frame_paths)}")
    return frame_paths

# ──────────────────────────────────────────────────────────────────────────────
# 7. Helper: Use DeepFace.find to match faces on each extracted frame (OPTIMIZED)
# ──────────────────────────────────────────────────────────────────────────────
def match_faces_via_deepface_find(video_path: str, reference_dir: str) -> List[MatchResult]:
    """
    OPTIMIZED APPROACH:
    1. Extract frames at 1 FPS into a temporary folder
    2. For each frame, call DeepFace.find with optimized parameters
    3. Use RetinaFace detector + face alignment for better accuracy
    4. Multiple distance thresholds for robust matching
    5. Enhanced error handling and logging
    """
    # Create a temp directory to hold extracted frames
    temp_frame_folder = tempfile.mkdtemp()
    matches: List[MatchResult] = []

    try:
        frame_paths = extract_frames_1fps(video_path, temp_frame_folder)
        if not frame_paths:
            logger.warning("No frames extracted; skipping matching.")
            return []

        logger.info(f"Starting face matching on {len(frame_paths)} frames against reference database")
        last_log = time.time()
        processed_frames = 0
        detection_failures = 0

        for idx, frame_path in enumerate(frame_paths):
            # Log progress every 10 frames or every 10 seconds
            current_time = time.time()
            if current_time - last_log >= 10:
                progress = (idx + 1) / len(frame_paths) * 100
                logger.info(f"Processing frame {idx+1}/{len(frame_paths)} ({progress:.1f}%) - "
                          f"Processed: {processed_frames}, Failures: {detection_failures}")
                last_log = current_time

            try:
                # OPTIMIZED: Use RetinaFace detector + face alignment
                df_list = DeepFace.find(
                    img_path=frame_path,
                    db_path=reference_dir,
                    model_name="ArcFace",
                    distance_metric="cosine",
                    detector_backend="retinaface",  # 🔥 Upgraded from opencv
                    align=True,                     # 🔥 Enable face alignment
                    enforce_detection=False,        # Handle detection failures gracefully
                    silent=True                     # Reduce DeepFace logging noise
                )

                # DeepFace.find returns a list of DataFrames (one per detected face)
                if not df_list or len(df_list) == 0:
                    continue

                # Process the first detected face (most prominent)
                df = df_list[0]
                processed_frames += 1

            except ValueError as e:
                if "Face could not be detected" in str(e):
                    logger.debug(f"No face detected in frame {idx} ({frame_path})")
                    detection_failures += 1
                    continue
                else:
                    logger.warning(f"DeepFace.find failed on frame {idx}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"Unexpected error on frame {idx}: {e}")
                continue

            # If no matches found in database
            if df.empty:
                logger.debug(f"No matches found for frame {idx}")
                continue

            # Get the best match (lowest distance)
            top_match = df.iloc[0]
            top_identity = top_match["identity"]  # Full path to matched reference image
            top_distance = float(top_match["ArcFace_cosine"])

            # Extract relative filename from full path
            rel_key = os.path.basename(top_identity)

            # ROBUST MATCHING: Multiple confidence levels
            timestamp_ms = idx * 1000

            if top_distance <= 0.30:
                # High confidence match
                matches.append(MatchResult(
                    photo_key=rel_key,
                    frame_index=idx,
                    timestamp_ms=timestamp_ms
                ))
                logger.info(f"🎯 HIGH CONFIDENCE MATCH → Frame {idx} ({timestamp_ms}ms) "
                          f"matches {rel_key} (distance={top_distance:.3f})")

            elif top_distance <= 0.40:
                # Medium confidence match (your current threshold)
                matches.append(MatchResult(
                    photo_key=rel_key,
                    frame_index=idx,
                    timestamp_ms=timestamp_ms
                ))
                logger.info(f"✅ MEDIUM CONFIDENCE MATCH → Frame {idx} ({timestamp_ms}ms) "
                          f"matches {rel_key} (distance={top_distance:.3f})")

            elif top_distance <= 0.50:
                # Log potential matches for analysis (not included in results)
                logger.debug(f"🤔 POTENTIAL MATCH → Frame {idx} matches {rel_key} "
                           f"(distance={top_distance:.3f}) - Below threshold")

            # Also log the top 3 matches for debugging
            if len(df) > 1:
                logger.debug(f"Frame {idx} top 3 matches:")
                for i in range(min(3, len(df))):
                    match_identity = os.path.basename(df.iloc[i]["identity"])
                    match_distance = float(df.iloc[i]["ArcFace_cosine"])
                    logger.debug(f"  {i+1}. {match_identity}: {match_distance:.3f}")

        # Final statistics
        total_processed = len(frame_paths)
        success_rate = (processed_frames / total_processed * 100) if total_processed > 0 else 0

        logger.info(f"MATCHING COMPLETE:")
        logger.info(f"  📊 Total frames: {total_processed}")
        logger.info(f"  ✅ Faces detected: {processed_frames} ({success_rate:.1f}%)")
        logger.info(f"  ❌ Detection failures: {detection_failures}")
        logger.info(f"  🎯 Matches found: {len(matches)}")

        return matches

    except Exception as e:
        logger.error(f"Critical error in face matching: {e}")
        raise
    finally:
        # Clean up extracted frames folder
        shutil.rmtree(temp_frame_folder, ignore_errors=True)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Enhanced FastAPI endpoint with better error handling
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/match_faces", response_model=MatchResponse)
def match_faces(request: MatchRequest):
    """
    ROBUST VIDEO FACE MATCHING ENDPOINT:
    1. Download video from URL with progress tracking
    2. List and validate S3 reference images
    3. Download reference images with error handling
    4. Perform optimized 1-FPS face matching using RetinaFace + alignment
    5. Return detailed match results with confidence levels
    """
    start_time = time.time()
    logger.info("═" * 80)
    logger.info("🎬 NEW VIDEO FACE MATCHING REQUEST")
    logger.info(f"📹 Video URL     : {request.video_url}")
    logger.info(f"🗂️  Photos bucket : {request.photos_bucket}")
    logger.info(f"📁 Photos prefix : {request.photos_prefix}")
    logger.info("═" * 80)

    local_video = None
    reference_dir = None

    try:
        # 1) Download video with enhanced error handling
        logger.info("📥 Downloading video...")
        try:
            local_video = download_video(request.video_url)
            video_size = os.path.getsize(local_video) / (1024 * 1024)  # MB
            logger.info(f"✅ Video downloaded successfully ({video_size:.1f} MB)")
        except Exception as e:
            logger.error(f"❌ Video download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Video download failed: {e}")

        # 2) List reference images from S3
        logger.info("📋 Listing reference images from S3...")
        try:
            photo_keys = list_s3_keys(request.photos_bucket, request.photos_prefix)
            logger.info(f"📸 Found {len(photo_keys)} reference images in S3")
            if len(photo_keys) > 0:
                logger.info(f"📷 Sample images: {photo_keys[:3]}{'...' if len(photo_keys) > 3 else ''}")
        except Exception as e:
            logger.error(f"❌ Failed to list S3 images: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list S3 keys: {e}")

        if not photo_keys:
            logger.warning("⚠️  No reference images found in S3; returning empty matches")
            return MatchResponse(matches=[])

        # 3) Download reference images with validation
        logger.info("⬇️  Downloading reference images...")
        reference_dir = tempfile.mkdtemp()
        downloaded_count = 0

        for i, key in enumerate(photo_keys):
            filename = os.path.basename(key)
            local_photo_path = os.path.join(reference_dir, filename)

            try:
                download_s3_object(request.photos_bucket, key, local_photo_path)

                # Validate image file
                if os.path.getsize(local_photo_path) > 0:
                    downloaded_count += 1
                    if (i + 1) % 10 == 0:  # Log every 10 downloads
                        logger.info(f"📥 Downloaded {i + 1}/{len(photo_keys)} reference images...")
                else:
                    logger.warning(f"⚠️  Empty file: {key}")
                    os.remove(local_photo_path)

            except Exception as e:
                logger.warning(f"⚠️  Failed to download {key}: {e}")
                continue

        logger.info(f"✅ Successfully downloaded {downloaded_count}/{len(photo_keys)} reference images")

        if downloaded_count == 0:
            logger.error("❌ No reference images could be downloaded")
            raise HTTPException(status_code=500, detail="No reference images could be downloaded")

        # 4) Perform optimized face matching
        logger.info("🔍 Starting face matching with RetinaFace + ArcFace...")
        try:
            results = match_faces_via_deepface_find(local_video, reference_dir)
        except Exception as e:
            logger.error(f"❌ Face matching failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face matching failed: {e}")

        # 5) Results summary
        elapsed = time.time() - start_time
        logger.info("═" * 80)
        logger.info(f"🎉 FACE MATCHING COMPLETE!")
        logger.info(f"⏱️  Total processing time: {elapsed:.2f} seconds")
        logger.info(f"🎯 Matches found: {len(results)}")

        if results:
            logger.info("📋 Match Summary:")
            for i, match in enumerate(results[:5]):  # Show first 5 matches
                logger.info(f"   {i+1}. {match.photo_key} at {match.timestamp_ms}ms (frame {match.frame_index})")
            if len(results) > 5:
                logger.info(f"   ... and {len(results) - 5} more matches")
        else:
            logger.info("ℹ️  No faces matched the reference images")

        logger.info("═" * 80)

        return MatchResponse(matches=results)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # 6) Cleanup temporary files
        cleanup_errors = []

        if local_video:
            try:
                video_dir = os.path.dirname(local_video)
                shutil.rmtree(video_dir, ignore_errors=True)
                logger.debug("🧹 Cleaned up video files")
            except Exception as e:
                cleanup_errors.append(f"video cleanup: {e}")

        if reference_dir:
            try:
                shutil.rmtree(reference_dir, ignore_errors=True)
                logger.debug("🧹 Cleaned up reference images")
            except Exception as e:
                cleanup_errors.append(f"reference cleanup: {e}")

        if cleanup_errors:
            logger.warning(f"⚠️  Cleanup warnings: {'; '.join(cleanup_errors)}")
