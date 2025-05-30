#!/usr/bin/env bash
set -euo pipefail

# ensure we’re in the project root
cd "$(dirname "$0")"

# name of our screen session
SCREEN_NAME="deepseekvl2_yolo"

# full command to run inside screen
CMD="source venv/bin/activate && \
pip install -r requirements_yolo.txt && \
uvicorn yolo_api:app --host 0.0.0.0 --port 8002"

# kill any existing session
if screen -list | grep -q "\.${SCREEN_NAME}[[:space:]]"; then
  echo "Stopping existing session: $SCREEN_NAME"
  screen -S "$SCREEN_NAME" -X quit
  sleep 1
fi

# start new detached screen
echo "Starting session: $SCREEN_NAME"
screen -dmS "$SCREEN_NAME" bash -lc "$CMD"

echo "✅ YOLO service is now running in screen '$SCREEN_NAME'."
