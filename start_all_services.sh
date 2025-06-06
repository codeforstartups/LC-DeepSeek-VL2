#!/usr/bin/env bash
set -euo pipefail

# 1️⃣ Go to this script's own dir (LC-DeepSeek-VL2)
cd "$(dirname "$0")"

# 2️⃣ Map each screen → startup command
declare -A SERVICES=(
  [deepseekvl2-video]="\
source venv/bin/activate && \
pip install -r requirements_video.txt && \
export AWS_S3_BUCKET=vision-app-prod-uploads && \
uvicorn video_api:app --host 0.0.0.0 --port 8001\
"
  [deepseekvl2_yolo]="\
source venv/bin/activate && \
pip install -r requirements_yolo.txt && \
uvicorn yolo_api:app --host 0.0.0.0 --port 8002\
"
  [deepseekvl2_medical]="\
source venv/bin/activate && \
uvicorn medical_imaging_api:app --host 0.0.0.0 --port 8000\
"
  [face_recognition]="\
source venv/bin/activate && \
pip install -r requirements_face_recognition.txt && \
uvicorn face_recognition:app --host 0.0.0.0 --port 8004 --reload\
"
  [deepseekvl2_models]="\
chmod +x start-ollama.sh && \
docker compose -f ollama-compose.yml up\
"
  [vision-backend]="\
cd .. && cd langchain-frontend-vision/backend && \
pip install -r requirements.txt && \
uvicorn main:app --reload --host 0.0.0.0 --port 3000\
"
  [vision-frontend]="\
cd .. && cd langchain-frontend-vision && \
npm install && \
npm run build && \
npm run preview\
"
  [vidoe_object_detection_deepseekvl2]="\
source object_detection/bin/activate && \
pip install -r requirements__object_detection.txt && \
uvicorn video_object_detection:app --host 0.0.0.0 --port 8005 --reload\
"
)

# 3️⃣ Helper to restart one service
restart_service() {
  local name=$1 cmd=$2

  # Stop existing session
  if screen -list | grep -q "\.${name}[[:space:]]"; then
    echo "⏹ Stopping existing session: $name"
    screen -S "$name" -X quit
    sleep 1
  fi

  # Start fresh detached screen
  echo "▶️ Starting session: $name"
  screen -dmS "$name" bash -lc "$cmd"
}

# 4️⃣ Loop through all
for name in "${!SERVICES[@]}"; do
  restart_service "$name" "${SERVICES[$name]}"
done

echo "✅ All services launched in their own screen sessions."
