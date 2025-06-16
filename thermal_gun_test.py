# Step 0: Required installations
# pip install roboflow ultralytics opencv-python numpy

import os
from datetime import datetime
from roboflow import Roboflow  # Correct import; RoboflowException is not exposed at top level
from ultralytics import YOLO
import torch

# ------------------------------------------------------------------
# Paste your private API key here, or set it as an environment variable:
# e.g., export ROBOFLOW_API_KEY="ELd5u4EvUH6cm8YyzIPf"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "ELd5u4EvUH6cm8YyzIPf")
# ------------------------------------------------------------------

# Step 1: Log into Roboflow and download dataset
print(f"[{datetime.now()}] Logging into Roboflow...")
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)  # Authenticate with Roboflow
    # Use the exact project slug: "precise-thermalpistol"
    project = rf.workspace("gun-detection-using-thermal-camera")\
                .project("precise-thermalpistol")  # Corrected slug :contentReference[oaicite:9]{index=9}
    version = project.version(1)
    print(f"[{datetime.now()}] Downloading Precise-thermalPistol v1 for YOLOv8...")
    dataset = version.download("yolov8")  # Downloads data.yaml + images/labels in YOLOv8 format
    print(f"[{datetime.now()}] Dataset download complete. Location: {dataset.location}")
except Exception as e:
    # Catch errors such as invalid slug or missing version :contentReference[oaicite:11]{index=11}
    print(f"[{datetime.now()}] ERROR: Unable to download dataset – {e}")
    exit(1)

# Step 2: Load a pre-trained YOLOv8 model
# You can switch to 'yolov8s.pt' or 'yolov8m.pt' if you have more GPU memory.
print(f"[{datetime.now()}] Loading pre-trained YOLOv8n model...")
model = YOLO('yolov8n.pt')  # COCO-pretrained YOLOv8n
print(f"[{datetime.now()}] Model loaded. Device: {model.device}")

# Step 3: Confirm dataset structure
data_yaml = os.path.join(dataset.location, "data.yaml")
if not os.path.isfile(data_yaml):
    print(f"[{datetime.now()}] ERROR: data.yaml not found in {dataset.location}")
    exit(1)  # Exit if data.yaml is missing

# Step 4: Train the model on the new dataset
print(f"[{datetime.now()}] Starting training (50 epochs @ 640×640)...")

# Check device from environment variable
device = os.getenv("TORCH_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[{datetime.now()}] Using device: {device}")

results = model.train(
    data=data_yaml,       # Path to the downloaded data.yaml
    epochs=50,            # Number of epochs (monitor metrics for early stopping)
    imgsz=640,            # Image size 640×640
    batch=8,
    device=device,        # Use environment-configured device
    project='MyTrainedModels',
    name='ThermalPistol_v1',
    exist_ok=True         # Overwrite existing folder if present
)
print(f"[{datetime.now()}] Training finished. Checkpoint: "
      f"MyTrainedModels/ThermalPistol_v1/weights/best.pt")

# Step 5: Quick validation on a sample frame (optional)
# Many Roboflow downloads include a 'test' folder; adjust if your structure differs.
sample_frame = os.path.join(dataset.location, "test", "sample.jpg")
if os.path.isfile(sample_frame):
    print(f"[{datetime.now()}] Running quick inference on sample frame...")
    res = model(sample_frame)[0]  # Inference on a single image
    res.show()  # Display bounding box if a GUI is available
else:
    print(f"[{datetime.now()}] No sample frame found for quick inference.")

# Step 6: Final confirmation
print(f"\n--- SCRIPT COMPLETE ---")
print(f"[{datetime.now()}] Trained model available at:")
print(f"    MyTrainedModels/ThermalPistol_v1/weights/best.pt")
