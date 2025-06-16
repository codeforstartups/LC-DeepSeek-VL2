import os
import tempfile
import shutil
import time
import uuid
import boto3
import cv2
import requests
import logging
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
from typing import List, Optional

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

# Check if GPU usage is disabled via environment variable
if os.getenv("TF_FORCE_GPU_ALLOW_GROWTH") == "false":
    # Force CPU-only execution
    tf.config.set_visible_devices([], 'GPU')
    logger.info("TensorFlow forced to CPU-only mode")
else:
    # Enable GPU memory growth if available
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"TensorFlow GPU memory growth enabled for {len(gpus)} GPUs")
    else:
        logger.info("No GPUs detected, running on CPU")

# ──────────────────────────────────────────────────────────────────────────────
# 1. FastAPI app initialization with CORS
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enhanced Face Recognition API",
    description="Advanced video face recognition with comprehensive confidence analysis, facial attributes, and anti-spoofing detection."
)

# Add CORS middleware to allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. ENHANCED Pydantic models with confidence parameters
# ──────────────────────────────────────────────────────────────────────────────
class FaceQuality(BaseModel):
    """Face quality assessment metrics"""
    face_confidence: float     # Detector confidence (0-1)
    face_size: int            # Face area in pixels
    is_blurry: bool           # Whether face appears blurry
    face_angle: str           # Face orientation (frontal, profile, etc.)
    brightness_score: float   # Face brightness (0-1)

class FaceAttributes(BaseModel):
    """Optional facial attribute analysis"""
    age: int                  # Estimated age
    gender: str               # Male/Female
    emotion: str              # Dominant emotion
    race: str                 # Dominant ethnicity
    age_confidence: float     # Age prediction confidence
    gender_confidence: float  # Gender prediction confidence
    emotion_confidence: float # Emotion prediction confidence
    race_confidence: float    # Race prediction confidence

class MatchRequest(BaseModel):
    video_url: str       # Public HTTPS video URL
    photos_bucket: str   # S3 bucket containing reference images
    photos_prefix: str   # Prefix (folder) under which reference images reside

    # 🔥 NEW: Configurable confidence parameters
    high_confidence_threshold: float = 0.30    # High confidence matches
    medium_confidence_threshold: float = 0.40  # Medium confidence matches
    low_confidence_threshold: float = 0.50     # Low confidence matches (logged only)
    include_face_attributes: bool = False       # Age, gender, emotion analysis
    include_anti_spoofing: bool = False        # Detect real vs fake faces
    include_face_quality: bool = True          # Face quality metrics
    min_face_size: int = 30                    # Minimum face size in pixels
    max_faces_per_frame: int = 10              # Limit faces processed per frame

class MatchResult(BaseModel):
    photo_key: str           # S3 key of matched reference image
    frame_index: int         # Which second/frame (0-based) matched
    timestamp_ms: int        # Millisecond timestamp for that frame
    face_index: int          # Which face in the frame (0-based)

    # 🔥 ENHANCED: Detailed confidence metrics
    distance: float          # Raw distance score
    confidence_level: str    # "HIGH", "MEDIUM", "LOW"
    confidence_score: float  # Normalized confidence (0-1)
    similarity_percentage: float  # Human-readable similarity %

    # 🔥 NEW: Face quality and attributes
    face_quality: FaceQuality
    face_attributes: Optional[FaceAttributes] = None  # Optional
    is_real_face: Optional[bool] = None              # Anti-spoofing result

    # 🔥 NEW: Additional match metadata
    all_matches_count: int   # Total potential matches found for this face
    rank: int               # Rank of this match (1=best, 2=second best, etc.)

class MatchResponse(BaseModel):
    matches: List[MatchResult]
    # 🔥 NEW: Processing statistics
    total_frames_processed: int
    total_faces_detected: int
    processing_time_seconds: float
    high_confidence_matches: int
    medium_confidence_matches: int

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper functions for enhanced analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_face_quality(face_region, face_confidence: float = 1.0) -> FaceQuality:
    """
    Analyze face quality metrics including size, blur, brightness, etc.
    """
    try:
        # Calculate face size
        height, width = face_region.shape[:2]
        face_size = height * width

        # Detect blur using Laplacian variance
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 100  # Threshold for blur detection

        # Calculate brightness
        brightness = np.mean(gray) / 255.0

        # Determine face angle (simplified)
        face_angle = "frontal"  # Could be enhanced with landmark detection
        if width < height * 0.7:
            face_angle = "profile"

        return FaceQuality(
            face_confidence=face_confidence,
            face_size=face_size,
            is_blurry=is_blurry,
            face_angle=face_angle,
            brightness_score=brightness
        )
    except Exception as e:
        logger.warning(f"Face quality analysis failed: {e}")
        return FaceQuality(
            face_confidence=face_confidence,
            face_size=0,
            is_blurry=False,
            face_angle="unknown",
            brightness_score=0.5
        )

def analyze_face_attributes(face_path: str) -> FaceAttributes:
    """
    Analyze facial attributes: age, gender, emotion, race
    """
    try:
        # Use DeepFace to analyze facial attributes
        analysis = DeepFace.analyze(
            img_path=face_path,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False,
            silent=True
        )

        # Handle both single face and multiple faces
        if isinstance(analysis, list):
            analysis = analysis[0]  # Take first face

        return FaceAttributes(
            age=int(analysis.get('age', 0)),
            gender="Female" if analysis.get('gender', {}).get('Woman', 0) > analysis.get('gender', {}).get('Man', 0) else "Male",
            emotion=max(analysis.get('emotion', {}), key=analysis.get('emotion', {}).get) if analysis.get('emotion') else "unknown",
            race=max(analysis.get('race', {}), key=analysis.get('race', {}).get) if analysis.get('race') else "unknown",
            age_confidence=1.0,  # DeepFace doesn't provide confidence for age
            gender_confidence=max(analysis.get('gender', {}).values()) if analysis.get('gender') else 0.0,
            emotion_confidence=max(analysis.get('emotion', {}).values()) if analysis.get('emotion') else 0.0,
            race_confidence=max(analysis.get('race', {}).values()) if analysis.get('race') else 0.0
        )
    except Exception as e:
        logger.warning(f"Face attribute analysis failed: {e}")
        return FaceAttributes(
            age=0,
            gender="unknown",
            emotion="unknown",
            race="unknown",
            age_confidence=0.0,
            gender_confidence=0.0,
            emotion_confidence=0.0,
            race_confidence=0.0
        )

def check_face_anti_spoofing(face_path: str) -> bool:
    """
    Check if face is real or fake using anti-spoofing
    """
    try:
        face_objs = DeepFace.extract_faces(
            img_path=face_path,
            anti_spoofing=True,
            enforce_detection=False
        )

        if face_objs and len(face_objs) > 0:
            return face_objs[0].get("is_real", True)
        return True
    except Exception as e:
        logger.warning(f"Anti-spoofing check failed: {e}")
        return True  # Default to True if check fails

def calculate_confidence_metrics(distance: float, high_thresh: float, medium_thresh: float, low_thresh: float):
    """
    Calculate comprehensive confidence metrics from distance
    """
    # Determine confidence level
    if distance <= high_thresh:
        confidence_level = "HIGH"
        confidence_score = 1.0 - (distance / high_thresh) * 0.3  # 0.7-1.0 range
    elif distance <= medium_thresh:
        confidence_level = "MEDIUM"
        confidence_score = 0.7 - ((distance - high_thresh) / (medium_thresh - high_thresh)) * 0.3  # 0.4-0.7 range
    elif distance <= low_thresh:
        confidence_level = "LOW"
        confidence_score = 0.4 - ((distance - medium_thresh) / (low_thresh - medium_thresh)) * 0.4  # 0.0-0.4 range
    else:
        confidence_level = "VERY_LOW"
        confidence_score = 0.0

    # Calculate similarity percentage (inverse of distance)
    similarity_percentage = max(0, (1 - distance) * 100)

    return confidence_level, confidence_score, similarity_percentage

# ──────────────────────────────────────────────────────────────────────────────
# 4. Helper: Create UUID-based temporary directory
# ──────────────────────────────────────────────────────────────────────────────
def create_request_temp_dir() -> str:
    """
    Create a unique temporary directory for this request using UUID.
    Returns the path to the temp directory.
    """
    request_id = str(uuid.uuid4())
    temp_base = tempfile.gettempdir()
    request_temp_dir = os.path.join(temp_base, f"face_recognition_{request_id}")
    os.makedirs(request_temp_dir, exist_ok=True)
    logger.info(f"🆔 Created request temp directory: {request_temp_dir}")
    return request_temp_dir

# ──────────────────────────────────────────────────────────────────────────────
# 5. Helper: Download video from a URL to request-specific temp folder
# ──────────────────────────────────────────────────────────────────────────────
def download_video(video_url: str, request_temp_dir: str) -> str:
    """
    Download the given video URL into the request-specific temp directory.
    """
    logger.info(f"📥 Downloading video from: {video_url}")
    local_video_path = os.path.join(request_temp_dir, "input_video.mp4")

    try:
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(local_video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Video download failed: {e}")

    logger.info(f"✅ Video saved to: {local_video_path}")
    return local_video_path

# ──────────────────────────────────────────────────────────────────────────────
# 6. Helper: List all object keys under a given S3 prefix
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
# 7. Helper: Download a single S3 object to request-specific folder
# ──────────────────────────────────────────────────────────────────────────────
def download_s3_object(bucket: str, key: str, local_path: str):
    """
    Download the S3 object at 'key' in 'bucket' into 'local_path'. Creates directories as needed.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    boto3.client("s3").download_file(Bucket=bucket, Key=key, Filename=local_path)

# ──────────────────────────────────────────────────────────────────────────────
# 8. Helper: Extract frames at 1 FPS from video
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

    logger.info(f"🎬 Video Duration: {duration_s:.2f}s @ {fps:.2f} FPS; extracting ~{int(duration_s)} frames at 1 FPS")

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
            logger.info(f"📸 Extracted frame at {sec}s → {frame_filename}")

        sec += 1

    cap.release()
    logger.info(f"✅ Total frames extracted: {len(frame_paths)}")
    return frame_paths

# ──────────────────────────────────────────────────────────────────────────────
# 9. ENHANCED Helper: Use DeepFace.find with comprehensive confidence analysis
# ──────────────────────────────────────────────────────────────────────────────
def match_faces_via_deepface_find(video_path: str, reference_dir: str, request_temp_dir: str, config: MatchRequest) -> tuple[List[MatchResult], dict]:
    """
    ENHANCED APPROACH with comprehensive confidence analysis:
    1. Extract frames at 1 FPS into request-specific temp folder
    2. For each frame, call DeepFace.find with optimized parameters
    3. Analyze face quality, attributes, and anti-spoofing
    4. Calculate detailed confidence metrics
    5. Enhanced error handling and logging
    """
    # Create frames folder within request temp directory
    frames_folder = os.path.join(request_temp_dir, "frames")
    matches: List[MatchResult] = []
    stats = {
        'total_faces_detected': 0,
        'processed_frames': 0,
        'detection_failures': 0
    }

    try:
        frame_paths = extract_frames_1fps(video_path, frames_folder)
        if not frame_paths:
            logger.warning("⚠️  No frames extracted; skipping matching.")
            return [], stats

        logger.info(f"🔍 Starting ENHANCED face matching on {len(frame_paths)} frames")
        logger.info(f"📊 Confidence thresholds: HIGH≤{config.high_confidence_threshold}, MEDIUM≤{config.medium_confidence_threshold}, LOW≤{config.low_confidence_threshold}")
        logger.info(f"🎛️  Features: Attributes={config.include_face_attributes}, Anti-spoofing={config.include_anti_spoofing}, Quality={config.include_face_quality}")

        last_log = time.time()

        for idx, frame_path in enumerate(frame_paths):
            # Log progress every 10 frames or every 10 seconds
            current_time = time.time()
            if current_time - last_log >= 10:
                progress = (idx + 1) / len(frame_paths) * 100
                logger.info(f"⏳ Processing frame {idx+1}/{len(frame_paths)} ({progress:.1f}%) - "
                          f"Processed: {stats['processed_frames']}, Faces: {stats['total_faces_detected']}, Failures: {stats['detection_failures']}")
                last_log = current_time

            try:
                # OPTIMIZED: Use RetinaFace detector + face alignment
                df_list = DeepFace.find(
                    img_path=frame_path,
                    db_path=reference_dir,
                    model_name="ArcFace",
                    distance_metric="cosine",
                    detector_backend="retinaface",
                    align=True,
                    enforce_detection=False,
                    silent=True
                )

                # DeepFace.find returns a list of DataFrames (one per detected face)
                if not df_list or len(df_list) == 0:
                    continue

                stats['processed_frames'] += 1
                frame_faces_count = len(df_list)
                stats['total_faces_detected'] += frame_faces_count

                # Limit number of faces processed per frame
                max_faces = min(frame_faces_count, config.max_faces_per_frame)
                if frame_faces_count > config.max_faces_per_frame:
                    logger.info(f"⚠️  Frame {idx}: Processing {max_faces}/{frame_faces_count} faces (limited by max_faces_per_frame)")

                # DEBUG: Log DataFrame structure for troubleshooting
                if idx == 0:  # Log only for first frame to avoid spam
                    logger.info(f"🐛 DEBUG: Number of faces detected: {frame_faces_count}")
                    if df_list[0] is not None and not df_list[0].empty:
                        logger.info(f"🐛 DEBUG: DataFrame columns: {list(df_list[0].columns)}")
                        logger.info(f"🐛 DEBUG: DataFrame shape: {df_list[0].shape}")

                # 🔥 ENHANCED: Process faces with comprehensive analysis
                for face_idx in range(max_faces):
                    df = df_list[face_idx]
                    logger.debug(f"👤 Processing face {face_idx + 1}/{max_faces} in frame {idx}")

                    # If no matches found in database for this face
                    if df.empty:
                        logger.debug(f"❌ No matches found for face {face_idx + 1} in frame {idx}")
                        continue

                    # Extract face region for quality analysis
                    face_region = None
                    try:
                        faces = DeepFace.extract_faces(
                            img_path=frame_path,
                            detector_backend="retinaface",
                            enforce_detection=False
                        )
                        if faces and len(faces) > face_idx:
                            face_region = (faces[face_idx]['face'] * 255).astype('uint8')
                    except Exception as e:
                        logger.debug(f"Face extraction failed for quality analysis: {e}")

                    # Get all matches for this face (sorted by distance)
                    distance_columns = [col for col in df.columns if 'cosine' in col.lower() or 'distance' in col.lower()]
                    if distance_columns:
                        distance_col = distance_columns[0]
                        all_matches = df.sort_values(by=distance_col)
                    else:
                        possible_cols = ['cosine', 'ArcFace_cosine', 'distance', 'similarity']
                        distance_col = None
                        for col in possible_cols:
                            if col in df.columns:
                                distance_col = col
                                break
                        if not distance_col:
                            logger.error(f"❌ Cannot find distance column. Columns: {list(df.columns)}")
                            continue
                        all_matches = df.sort_values(by=distance_col)

                    # Process top matches up to low confidence threshold
                    for rank, (match_idx, match_row) in enumerate(all_matches.iterrows(), 1):
                        distance = float(match_row[distance_col])

                        # Skip matches beyond low confidence threshold
                        if distance > config.low_confidence_threshold:
                            break

                        top_identity = match_row["identity"]
                        rel_key = os.path.basename(top_identity)

                        # Calculate comprehensive confidence metrics
                        confidence_level, confidence_score, similarity_percentage = calculate_confidence_metrics(
                            distance, config.high_confidence_threshold,
                            config.medium_confidence_threshold, config.low_confidence_threshold
                        )

                        # Only add to results if above medium confidence
                        if distance <= config.medium_confidence_threshold:
                            # 🔥 ENHANCED: Face quality analysis
                            face_quality = FaceQuality(
                                face_confidence=1.0,
                                face_size=0,
                                is_blurry=False,
                                face_angle="unknown",
                                brightness_score=0.5
                            )

                            if config.include_face_quality and face_region is not None:
                                face_quality = analyze_face_quality(face_region)

                                # Skip low quality faces if configured
                                if face_quality.face_size < config.min_face_size:
                                    logger.debug(f"⚠️  Skipping small face: {face_quality.face_size} < {config.min_face_size}")
                                    continue

                            # 🔥 ENHANCED: Facial attributes analysis
                            face_attributes = None
                            if config.include_face_attributes:
                                face_attributes = analyze_face_attributes(frame_path)

                            # 🔥 ENHANCED: Anti-spoofing check
                            is_real_face = None
                            if config.include_anti_spoofing:
                                is_real_face = check_face_anti_spoofing(frame_path)
                                if not is_real_face:
                                    logger.warning(f"🚫 Fake face detected in frame {idx}, face {face_idx + 1}")

                            # Create enhanced match result
                            match_result = MatchResult(
                                photo_key=rel_key,
                                frame_index=idx,
                                timestamp_ms=idx * 1000,
                                face_index=face_idx,
                                distance=distance,
                                confidence_level=confidence_level,
                                confidence_score=confidence_score,
                                similarity_percentage=similarity_percentage,
                                face_quality=face_quality,
                                face_attributes=face_attributes,
                                is_real_face=is_real_face,
                                all_matches_count=len(all_matches),
                                rank=rank
                            )

                            matches.append(match_result)

                            # Enhanced logging with confidence details
                            emoji = "🎯" if confidence_level == "HIGH" else "✅"
                            logger.info(f"{emoji} {confidence_level} MATCH → Frame {idx} Face {face_idx + 1} "
                                      f"matches {rel_key} (distance={distance:.3f}, "
                                      f"similarity={similarity_percentage:.1f}%, rank={rank})")

                        # Log potential matches for analysis
                        elif distance <= config.low_confidence_threshold:
                            logger.debug(f"🤔 LOW CONFIDENCE → Frame {idx} Face {face_idx + 1} matches {rel_key} "
                                       f"(distance={distance:.3f}, similarity={similarity_percentage:.1f}%)")

            except ValueError as e:
                if "Face could not be detected" in str(e):
                    logger.debug(f"😐 No face detected in frame {idx}")
                    stats['detection_failures'] += 1
                    continue
                else:
                    logger.warning(f"⚠️  DeepFace.find failed on frame {idx}: {e}")
                    continue
            except Exception as e:
                logger.warning(f"💥 Unexpected error on frame {idx}: {e}")
                continue

        return matches, stats

    except Exception as e:
        logger.error(f"💥 Critical error in enhanced face matching: {e}")
        raise

# ──────────────────────────────────────────────────────────────────────────────
# 10. ENHANCED FastAPI endpoint with comprehensive confidence analysis
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/match_faces", response_model=MatchResponse)
def match_faces(request: MatchRequest):
    """
    ENHANCED VIDEO FACE MATCHING ENDPOINT WITH COMPREHENSIVE CONFIDENCE ANALYSIS:
    1. Create UUID-based temp directory for this request
    2. Download video and reference images to isolated folders
    3. Perform optimized 1-FPS face matching with detailed confidence metrics
    4. Analyze face quality, attributes, and anti-spoofing (optional)
    5. Return comprehensive match results with confidence levels
    6. Guaranteed cleanup of all temp files
    """
    start_time = time.time()
    request_temp_dir = None

    # Generate unique request ID for logging
    request_id = str(uuid.uuid4())[:8]

    logger.info("═" * 90)
    logger.info(f"🎬 NEW ENHANCED VIDEO FACE MATCHING REQUEST [ID: {request_id}]")
    logger.info(f"📹 Video URL     : {request.video_url}")
    logger.info(f"🗂️  Photos bucket : {request.photos_bucket}")
    logger.info(f"📁 Photos prefix : {request.photos_prefix}")
    logger.info(f"🎯 Confidence    : HIGH≤{request.high_confidence_threshold}, MED≤{request.medium_confidence_threshold}, LOW≤{request.low_confidence_threshold}")
    logger.info(f"🔧 Features      : Attr={request.include_face_attributes}, Anti-spoof={request.include_anti_spoofing}, Quality={request.include_face_quality}")
    logger.info("═" * 90)

    try:
        # 1) Create UUID-based temporary directory for this request
        request_temp_dir = create_request_temp_dir()
        logger.info(f"🆔 Request temp directory: {request_temp_dir}")

        # 2) Download video with enhanced error handling
        logger.info("📥 Downloading video...")
        try:
            local_video = download_video(request.video_url, request_temp_dir)
            video_size = os.path.getsize(local_video) / (1024 * 1024)  # MB
            logger.info(f"✅ Video downloaded successfully ({video_size:.1f} MB)")
        except Exception as e:
            logger.error(f"❌ Video download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Video download failed: {e}")

        # 3) List reference images from S3
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
            return MatchResponse(
                matches=[],
                total_frames_processed=0,
                total_faces_detected=0,
                processing_time_seconds=time.time() - start_time,
                high_confidence_matches=0,
                medium_confidence_matches=0
            )

        # 4) Download reference images to isolated folder
        logger.info("⬇️  Downloading reference images...")
        reference_dir = os.path.join(request_temp_dir, "reference_photos")
        os.makedirs(reference_dir, exist_ok=True)
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

        # 5) Perform enhanced face matching with comprehensive confidence analysis
        logger.info("🔍 Starting ENHANCED face matching with comprehensive confidence analysis...")
        try:
            results, stats = match_faces_via_deepface_find(local_video, reference_dir, request_temp_dir, request)
        except Exception as e:
            logger.error(f"❌ Enhanced face matching failed: {e}")
            raise HTTPException(status_code=500, detail=f"Enhanced face matching failed: {e}")

        # 6) Calculate enhanced statistics
        elapsed = time.time() - start_time
        high_conf_matches = len([m for m in results if m.confidence_level == "HIGH"])
        medium_conf_matches = len([m for m in results if m.confidence_level == "MEDIUM"])

        # Enhanced final statistics
        total_processed = stats.get('processed_frames', 0)
        total_faces = stats.get('total_faces_detected', 0)
        avg_faces_per_frame = total_faces / total_processed if total_processed > 0 else 0

        logger.info("═" * 90)
        logger.info(f"🎉 ENHANCED FACE MATCHING COMPLETE! [ID: {request_id}]")
        logger.info(f"⏱️  Total processing time: {elapsed:.2f} seconds")
        logger.info(f"📊 Enhanced Statistics:")
        logger.info(f"  📊 Total frames processed: {total_processed}")
        logger.info(f"  👥 Total faces detected: {total_faces} ({avg_faces_per_frame:.1f} per frame)")
        logger.info(f"  ❌ Detection failures: {stats.get('detection_failures', 0)}")
        logger.info(f"  🎯 HIGH confidence matches: {high_conf_matches}")
        logger.info(f"  ✅ MEDIUM confidence matches: {medium_conf_matches}")
        logger.info(f"  📈 Total matches returned: {len(results)}")

        if results:
            logger.info("📋 Top Matches Summary:")
            for i, match in enumerate(results[:5]):  # Show first 5 matches
                logger.info(f"   {i+1}. {match.photo_key} at {match.timestamp_ms}ms "
                          f"({match.confidence_level}, {match.similarity_percentage:.1f}%)")
            if len(results) > 5:
                logger.info(f"   ... and {len(results) - 5} more matches")
        else:
            logger.info("ℹ️  No faces matched the reference images with sufficient confidence")

        logger.info("═" * 90)

        return MatchResponse(
            matches=results,
            total_frames_processed=total_processed,
            total_faces_detected=total_faces,
            processing_time_seconds=elapsed,
            high_confidence_matches=high_conf_matches,
            medium_confidence_matches=medium_conf_matches
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error in enhanced request [{request_id}]: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # 7) GUARANTEED CLEANUP: Always clean up the entire request temp directory
        if request_temp_dir and os.path.exists(request_temp_dir):
            try:
                shutil.rmtree(request_temp_dir, ignore_errors=True)
                logger.info(f"🧹 Cleaned up request temp directory: {request_temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Cleanup warning for {request_temp_dir}: {e}")
        else:
            logger.debug("🧹 No temp directory to clean up")

# ──────────────────────────────────────────────────────────────────────────────
# 11. Health check endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "service": "enhanced_face_recognition_api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
