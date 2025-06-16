#!/usr/bin/env bash
set -euo pipefail

# 🚀 OPTIMIZED SERVICES STARTUP SCRIPT
# GPU Memory Optimization: Only medical imaging on GPU, everything else on CPU
# Separate virtual environments for memory tracking

# 1️⃣ Go to this script's own dir (LC-DeepSeek-VL2)
cd "$(dirname "$0")"

echo "🚀 Starting LC-DeepSeek-VL2 Services with Memory Optimization..."
echo "📊 Strategy: GPU T1 = Medical Imaging Only, CPU = Everything Else"
echo "═══════════════════════════════════════════════════════════════"

# 2️⃣ Create virtual environments if they don't exist
create_venv_if_missing() {
    local venv_name=$1
    local requirements_file=$2

    if [ ! -d "$venv_name" ]; then
        echo "📦 Creating virtual environment: $venv_name"
        python3 -m venv "$venv_name"
    fi

    echo "📥 Installing/updating requirements for $venv_name..."
    source "$venv_name/bin/activate"
    pip install --upgrade pip
    if [ -f "$requirements_file" ]; then
        pip install -r "$requirements_file"
    fi
    deactivate
}

# 3️⃣ Setup all virtual environments
echo "🔧 Setting up virtual environments..."

create_venv_if_missing "venv_video" "requirements_video.txt"
create_venv_if_missing "venv_yolo_cpu" "requirements_yolo.txt"
create_venv_if_missing "venv_medical_gpu" "requirements.txt"
create_venv_if_missing "venv_face_cpu" "requirements_face_recognition.txt"
create_venv_if_missing "venv_metal_cpu" "requirements_thermal_gun_api.txt"
create_venv_if_missing "venv_object_detection_cpu" "requirements__object_detection.txt"

echo "✅ All virtual environments ready!"

# 4️⃣ Map each screen → startup command with optimized GPU/CPU allocation
declare -A SERVICES=(

  # 🖥️ CPU-BASED SERVICES (Host Memory)
  [video_api_cpu]="\
source venv_video/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"\" && \
export DEEPSEEK_API_URL=\"http://localhost:8000/analyze/\" && \
export AWS_S3_BUCKET=vision-app-prod-uploads && \
echo '📹 Starting Video API on CPU...' && \
uvicorn video_api:app --host 0.0.0.0 --port 8001\
"

  [yolo_api_cpu]="\
source venv_yolo_cpu/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"\" && \
export TORCH_DEVICE=\"cpu\" && \
echo '🔍 Starting YOLO API on CPU...' && \
uvicorn yolo_api:app --host 0.0.0.0 --port 8002\
"

  [face_recognition_gpu]="\
source venv_face_cpu/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"0\" && \
export TF_FORCE_GPU_ALLOW_GROWTH=\"true\" && \
export TF_CPP_MIN_LOG_LEVEL=\"2\" && \
echo '👤 Starting Face Recognition on GPU T1...' && \
uvicorn face_recognition:app --host 0.0.0.0 --port 8004\
"

  [thermal_gun_cpu]="\
source venv_metal_cpu/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"\" && \
export TORCH_DEVICE=\"cpu\" && \
echo '🔫 Starting Thermal Gun Detection on CPU...' && \
uvicorn thermal_gun_detection_api:app --host 0.0.0.0 --port 8006\
"

  [object_detection_cpu]="\
source venv_object_detection_cpu/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"\" && \
export DEEPSEEK_API_URL=\"http://localhost:8000/analyze/\" && \
echo '🎯 Starting Object Detection on CPU...' && \
uvicorn video_object_detection:app --host 0.0.0.0 --port 8005\
"

  # 🚀 GPU-BASED SERVICES (T1 GPU Shared)
  [medical_imaging_gpu]="\
source venv_medical_gpu/bin/activate && \
export CUDA_VISIBLE_DEVICES=\"0\" && \
export PYTORCH_CUDA_ALLOC_CONF=\"max_split_size_mb:512\" && \
echo '🏥 Starting Medical Imaging on GPU T1...' && \
uvicorn medical_imaging_api:app --host 0.0.0.0 --port 8000 --workers 1\
"

  # 🐳 DOCKER SERVICES (Host Memory)
  [ollama_models_cpu]="\
export OLLAMA_NUM_GPU=0 && \
export OLLAMA_HOST=\"0.0.0.0:11434\" && \
echo '🤖 Starting Ollama Models on CPU...' && \
chmod +x start-ollama.sh && \
docker compose -f ollama-compose.yml up\
"

  # 🌐 FRONTEND/BACKEND (Host Memory)
  [vision_backend]="\
cd ../langchain-frontend-vision/backend && \
export CUDA_VISIBLE_DEVICES=\"\" && \
echo '🔗 Starting Vision Backend on CPU...' && \
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate && \
pip install -r requirements.txt && \
uvicorn main:app --host 0.0.0.0 --port 3000 --workers 4\
"

  [vision_frontend]="\
cd ../langchain-frontend-vision && \
echo '🎨 Starting Vision Frontend...' && \
npm install && \
npm run build && \
npm run preview\
"

)

# 5️⃣ Helper to restart one service with enhanced logging
restart_service() {
    local name=$1
    local cmd=$2

    # Stop existing session
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "⏹️  Stopping existing session: $name"
        screen -S "$name" -X quit
        sleep 2
    fi

    # Start fresh detached screen with logging
    echo "▶️  Starting service: $name"
    screen -dmS "$name" bash -lc "
        echo '🚀 Service $name starting at \$(date)...'
        $cmd
    "

    # Verify service started
    sleep 1
    if screen -list | grep -q "\.${name}[[:space:]]"; then
        echo "✅ Service $name started successfully"
    else
        echo "❌ Service $name failed to start"
    fi
}

# 6️⃣ Start services in optimized order
echo ""
echo "🚀 Starting services in optimal order..."
echo "📊 Memory Strategy: GPU=Medical Only, CPU=Everything Else"
echo ""

# Start GPU service first (medical imaging)
echo "🏥 Phase 1: Starting GPU-based services..."
restart_service "medical_imaging_gpu" "${SERVICES[medical_imaging_gpu]}"
sleep 3
restart_service "face_recognition_gpu" "${SERVICES[face_recognition_gpu]}"
sleep 3

# Start CPU-based core services
echo "🖥️  Phase 2: Starting CPU-based core services..."
restart_service "video_api_cpu" "${SERVICES[video_api_cpu]}"
sleep 2
restart_service "yolo_api_cpu" "${SERVICES[yolo_api_cpu]}"
sleep 2

# Start specialized CPU services
echo "🔧 Phase 3: Starting specialized CPU services..."
restart_service "thermal_gun_cpu" "${SERVICES[thermal_gun_cpu]}"
sleep 2
restart_service "object_detection_cpu" "${SERVICES[object_detection_cpu]}"
sleep 2

# Start support services
echo "🐳 Phase 4: Starting support services..."
restart_service "ollama_models_cpu" "${SERVICES[ollama_models_cpu]}"
sleep 3
restart_service "vision_backend" "${SERVICES[vision_backend]}"
sleep 2
restart_service "vision_frontend" "${SERVICES[vision_frontend]}"

echo ""
echo "✅ All services launched in optimized configuration!"
echo ""
echo "📊 Expected GPU Memory Usage:"
echo "  🏥 Medical Imaging (GPU): ~6-7GB"
echo "  👤 Face Recognition (GPU): ~1-2GB"
echo "  🖥️  All other services (CPU): Host RAM"
echo "  📊 Total GPU Usage: ~8-9GB (52-58% of 15.4GB)"
echo ""
echo "�� Monitor services:"
echo "  screen -list                    # List all services"
echo "  screen -r medical_imaging_gpu   # Attach to GPU service"
echo "  screen -r face_recognition_gpu  # Attach to face recognition"
echo "  nvidia-smi                      # Check GPU usage"
echo "  free -h                         # Check RAM usage"
echo ""
echo "🌐 Service URLs:"
echo "  Medical Imaging (GPU): http://localhost:8000"
echo "  Video API (CPU):       http://localhost:8001"
echo "  YOLO API (CPU):        http://localhost:8002"
echo "  Vision Backend (CPU):  http://localhost:3000"
echo "  Face Recognition (GPU): http://localhost:8004"
echo "  Object Detection (CPU): http://localhost:8005"
echo "  Thermal Gun (CPU):     http://localhost:8006"
echo "  Ollama (CPU):          http://localhost:11434"
echo ""
echo "🎯 Optimization Complete! GPU memory should be ~55% with Medical+Face on GPU"
