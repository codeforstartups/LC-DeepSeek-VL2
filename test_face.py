import os
import tensorflow as tf
from deepface import DeepFace

# ──────────────────────────────────────────────────────────────────────────────
# 0. Enable TensorFlow GPU Memory Growth (for g4dn.xlarge / T4 GPU)
# ──────────────────────────────────────────────────────────────────────────────
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Optionally limit to GPU:0 only
    # tf.config.set_visible_devices(gpus[0], "GPU")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Specify the image you want to test
# ──────────────────────────────────────────────────────────────────────────────
img_path = "images/multi_image_1.jpeg"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Attempt face embedding with enforce_detection=True (default)
# ──────────────────────────────────────────────────────────────────────────────
print("▶ Trying to compute embedding with enforce_detection=True (OpenCV)...")
try:
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        enforce_detection=True,
        detector_backend="opencv"
    )
    print("✅ Embedding succeeded (OpenCV backend): length =", len(embedding[0]["embedding"]))

except ValueError as ve:
    print("❌ Error:", ve)
    print("   → Switching to enforce_detection=False instead.\n")

    # ──────────────────────────────────────────────────────────────────────────────
    # 3. Retry with enforce_detection=False to bypass detection
    # ──────────────────────────────────────────────────────────────────────────────
    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=False,     # force embedding even if no face is found
            detector_backend="opencv"
        )
        print("⚠️  Embedding with enforce_detection=False: length =", len(embedding[0]["embedding"]))
        print("   → Note: Embedding may be invalid since no face was detected.")

    except Exception as e2:
        print("❌ Still failed with enforce_detection=False:", e2)
        print("   → You may want to try a different detector_backend or check the image.")

# ──────────────────────────────────────────────────────────────────────────────
# 4. (Optional) Try a stronger detector like MTCNN if a face should exist
# ──────────────────────────────────────────────────────────────────────────────
print("\n▶ Trying to compute embedding with detector_backend='mtcnn' (enforce_detection=True)...")
try:
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="ArcFace",
        enforce_detection=True,
        detector_backend="mtcnn"
    )
    print("✅ Embedding succeeded (MTCNN backend): length =", len(embedding[0]["embedding"]))

except Exception as e3:
    print("❌ Even MTCNN failed:", e3)
    print("   → Confirm that the image truly contains a front‐facing human face, or correct the image path.")
