# LC-DeepSeek-VL2 Services Technical Documentation

**Repository**: [https://github.com/codeforstartups/LC-DeepSeek-VL2](https://github.com/codeforstartups/LC-DeepSeek-VL2)

## 🚀 Quick Start Commands

```bash
# Start all services
./start_all_services_optimized.sh

# Stop all services
./stop_all_services.sh
```

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Design](#architecture--design)
3. [Service Details](#service-details)
4. [Deployment & Configuration](#deployment--configuration)
5. [API Reference](#api-reference)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)
 
---

## System Overview

### 🚀 LC-DeepSeek-VL2 Platform
The LC-DeepSeek-VL2 is a comprehensive AI/ML platform that provides 14 different services covering computer vision, natural language processing, database operations, and web interfaces. The system is optimized for GPU memory efficiency with intelligent resource allocation.

### 📊 Resource Allocation Strategy
- **GPU T1**: Medical Imaging + Face Recognition (8-9GB total)
- **CPU**: All other services use host RAM
- **Expected GPU Usage**: ~55% of 15.4GB total capacity

### 🎯 Key Features
- **Memory Optimization**: GPU memory limited to 85% usage
- **Progressive Fallback**: Image resizing for memory efficiency
- **Thread Safety**: Lock mechanisms for concurrent access
- **Error Handling**: Comprehensive retry and error recovery
- **Health Monitoring**: Health check endpoints for all services
- **CORS Support**: Cross-origin resource sharing enabled
- **Docker Integration**: Containerized services with health checks

---

## Architecture & Design

### 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LC-DeepSeek-VL2 Platform                   │
├─────────────────────────────────────────────────────────────────┤
│  🖥️  CPU-Based Services (Host RAM)                           │
│  ├── Video API (8001)                                         │
│  ├── YOLO API (8002)                                          │
│  ├── Object Detection API (8005)                              │
│  ├── Thermal Gun Detection API (8006)                         │
│  ├── Translation API (8007)                                   │
│  ├── Vanna Text-to-SQL API (8008)                            │
│  ├── Robust Text-to-SQL API (8009)                           │
│  ├── Embedding & Retrieval API (8010)                        │
│  ├── Vision Backend (3000)                                    │
│  └── Vision Frontend (Web)                                    │
├─────────────────────────────────────────────────────────────────┤
│  🚀 GPU-Based Services (T1 GPU - 8-9GB)                      │
│  ├── Medical Imaging API (8000) - 6-7GB                      │
│  └── Face Recognition API (8004) - 1-2GB                     │
├─────────────────────────────────────────────────────────────────┤
│  🐳 Docker Services                                            │
│  ├── MindsDB (1833)                                           │
│  ├── Ollama Models (11434)                                    │
│  ├── PostgreSQL (5432)                                        │
│  └── Qdrant Vector DB (6333)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 🔄 Service Dependencies

```
Medical Imaging API (8000)
├── DeepSeek-VL2 Model
└── GPU Memory Management

Face Recognition API (8004)
├── DeepFace + TensorFlow
├── S3 Integration
└── Anti-spoofing Detection

Video API (8001)
├── DeepSeek API (8000)
├── Ollama Chat API (11434)
└── S3 Upload

Object Detection API (8005)
├── DeepSeek API (8000)
└── Ollama Chat API (11434)

Vanna API (8008)
├── Ollama Models (11434)
├── Qdrant Vector DB (6333)
└── PostgreSQL (5432)

Robust SQL API (8009)
├── MindsDB (1833)
├── Ollama Models (11434)
└── PostgreSQL (5432)

Embedding API (8010)
├── Ollama Models (11434)
└── Qdrant Vector DB (6333)
```

---

## Service Details

### 🏥 1. Medical Imaging API
**File**: `medical_imaging_api.py`
**Port**: `8000`
**Memory**: ~6-7GB GPU
**Environment**: `venv_medical_gpu`

#### Purpose
Specialized medical imaging analysis using DeepSeek-VL2 for radiological scans (MRI, CT, X-Ray).

#### Key Features
- **Medical Analysis**: Detailed radiological findings and abnormality detection
- **OCR Capability**: Text extraction from medical documents
- **Object Detection**: General object detection capabilities
- **Memory Optimization**: FP16 precision, progressive image resizing
- **Medical Disclaimer**: Educational use warnings



#### API Endpoints
- `POST /analyze/` - Main analysis endpoint
- `GET /health` - Health check with GPU memory status
- `GET /gpu_memory` - GPU memory monitoring

#### Dependencies
```txt
torch==2.0.1
transformers==4.38.2
xformers>=0.0.21
timm>=0.9.16
accelerate
sentencepiece
attrdict
einops
gradio==3.48.0
```

---

### 👤 2. Face Recognition API
**File**: `face_recognition.py`
**Port**: `8004`
**Memory**: ~1-2GB GPU
**Environment**: `venv_face_cpu`

#### Purpose
Advanced face detection and recognition with comprehensive analysis capabilities.

#### Key Features
- **Face Detection**: Multi-face detection in videos/images
- **Anti-spoofing**: Real vs fake face detection
- **Face Quality Assessment**: Blur, angle, brightness analysis
- **Facial Attributes**: Age, gender, emotion, ethnicity
- **S3 Integration**: Reference image management
- **Multi-confidence Matching**: High/medium/low confidence thresholds



#### API Endpoints
- `POST /match_faces` - Main face matching endpoint
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn
boto3
opencv-python
deepface
numpy
requests
pydantic
tf-keras
```

---

### 📹 3. Video API
**File**: `video_api.py`
**Port**: `8001`
**Environment**: `venv_video`

#### Purpose
Video analysis and processing with frame-by-frame AI analysis.

#### Key Features
- **Frame Extraction**: 1 FPS frame extraction
- **DeepSeek Integration**: AI analysis of video frames
- **Ollama Summarization**: Video content summarization
- **S3 Upload**: Cloud storage integration
- **Multi-format Support**: MP4, AVI, MOV



#### API Endpoints
- `POST /analyze_video` - Video analysis endpoint
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn
httpx
opencv-python
pillow
python-multipart
requests
boto3
```

---

### 🔍 4. YOLO API
**File**: `yolo_api.py`
**Port**: `8002`
**Environment**: `venv_yolo_cpu`

#### Purpose
Real-time object detection using YOLOv8 model.

#### Key Features
- **YOLOv8 Model**: Latest YOLO architecture
- **Frame Processing**: 1 FPS frame extraction
- **Object Counting**: Aggregated object detection counts
- **Multi-format Support**: MP4, AVI, MOV
- **CPU Optimization**: Efficient CPU inference



#### API Endpoints
- `POST /detect_objects/` - Object detection endpoint
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn[standard]
python-multipart
ultralytics
opencv-python-headless
torch
requests
```

---

### 🎯 5. Object Detection API
**File**: `video_object_detection.py`
**Port**: `8005`
**Environment**: `venv_object_detection_cpu`

#### Purpose
Advanced object detection with DeepSeek-VL2 integration for detailed analysis.

#### Key Features
- **DeepSeek Integration**: AI-powered object analysis
- **Frame Processing**: 1 FPS extraction and analysis
- **Ollama Summarization**: Content summarization
- **Multi-format Support**: Various video formats
- **Detailed Analysis**: Comprehensive object descriptions



#### API Endpoints
- `POST /analyze_video` - Video object detection
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn[standard]
pydantic
httpx
opencv-python
Pillow
requests
```

---

### 🔫 6. Thermal Gun Detection API
**File**: `thermal_gun_detection_api.py`
**Port**: `8006`
**Environment**: `venv_metal_cpu`

#### Purpose
Specialized detection of thermal weapons using custom YOLO model.

#### Key Features
- **Custom YOLO Model**: Trained for thermal gun detection
- **Confidence Thresholds**: Configurable detection sensitivity
- **Video & Image Support**: Both video and image processing
- **Request Tracking**: UUID-based request tracking
- **Cleanup Management**: Automatic file cleanup



#### API Endpoints
- `POST /detect_thermal_guns_image/` - Image detection
- `POST /detect_thermal_guns/` - Video detection
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn[standard]
pydantic
httpx
opencv-python
Pillow
requests
ultralytics
torch
torchvision
pytube2
```

---

### 🗣️ 7. Translation API
**File**: `translation_api.py`
**Port**: `8007`
**Environment**: `venv_translation_cpu`

#### Purpose
Multi-language translation using NLLB-200 model.

#### Key Features
- **NLLB-200 Model**: 600M parameter multilingual model
- **20+ Languages**: Comprehensive language support
- **Thread Safety**: Lock mechanisms for concurrent access
- **GPU/CPU Support**: Flexible device selection
- **Language Mapping**: Simplified language codes



#### API Endpoints
- `POST /translate` - Translation endpoint
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn
torch
transformers
sentencepiece
accelerate
python-dotenv
```

---

### 🧠 8. Vanna Text-to-SQL API
**File**: `vanna_api.py`
**Port**: `8008`
**Environment**: `venv_vanna_api`

#### Purpose
Natural language to SQL conversion using Ollama and Qdrant.

#### Key Features
- **Ollama Integration**: Local LLM for SQL generation
- **Qdrant Vector Store**: Semantic search capabilities
- **PostgreSQL Connection**: Direct database access
- **Automated Training**: Schema-based training
- **Custom Examples**: Question-SQL pair training



#### API Endpoints
- `POST /sql_query` - SQL generation endpoint
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn[standard]
pandas
vanna[ollama,qdrant,postgres]
```

---

### 🤖 9. Robust Text-to-SQL API
**File**: `robust_text_to_sql_api.py`
**Port**: `8009`
**Environment**: `venv_robust_sql_api`

#### Purpose
Enhanced SQL generation using MindsDB and DuckDB-NSQL model.

#### Key Features
- **MindsDB Integration**: ML-powered SQL generation
- **DuckDB-NSQL Model**: Specialized SQL model
- **Remote/Local Support**: Flexible MindsDB deployment
- **Retry Mechanisms**: Robust error handling
- **PostgreSQL Connection**: Database integration



#### API Endpoints
- `POST /query` - SQL query generation
- `GET /health` - Health check
- `GET /examples` - Example queries
- `GET /status` - Service status

#### Dependencies
```txt
fastapi
uvicorn
mindsdb-sdk
pandas
sqlalchemy
psycopg2-binary
python-dotenv
pydantic
requests
```

---

### 🔍 10. Embedding & Retrieval API
**File**: `embedding_retrieval_api.py`
**Port**: `8010`
**Environment**: `venv_retrieval_cpu`

#### Purpose
Vector embeddings and semantic search using Ollama and Qdrant.

#### Key Features
- **Ollama Embeddings**: nomic-embed-text model
- **Qdrant Vector DB**: High-performance vector database
- **Text Chunking**: Intelligent text splitting
- **Thread Safety**: Concurrent access protection
- **Collection Management**: Dynamic collection creation



#### API Endpoints
- `POST /add` - Add documents to vector store
- `POST /search` - Semantic search
- `GET /health` - Health check

#### Dependencies
```txt
fastapi
uvicorn
langchain
langchain-community
qdrant-client
langchain-text-splitters
langchain-ollama
```

---

### 🐳 Docker Services

#### 11. MindsDB
**Port**: `1833`
**Purpose**: Machine learning database

**Features**:
- HTTP, PostgreSQL, MySQL APIs
- Host network mode
- Auto-restart capability

**Docker Configuration**:
```yaml
docker run -d --name mindsdb --network host \
  -e MINDSDB_APIS=http,postgres,mysql \
  mindsdb/mindsdb
```

#### 12. Ollama Models
**Port**: `11434`
**Purpose**: Local LLM models

**Features**:
- CPU-only inference
- Three specialized models:
  - `deepseek-r1:1.5b` (Text generation)
  - `nomic-embed-text` (Embeddings)
  - `duckdb-nsql:7b` (SQL generation)

**Docker Compose Configuration**:
```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    environment:
      OLLAMA_NUM_GPU: 0
      OLLAMA_NUM_GPU_LAYERS: 0
      OLLAMA_KEEP_ALIVE: "5m"
```

#### 13. PostgreSQL Database
**Port**: `5432`
**Purpose**: Primary database

**Features**:
- Database: `langchain_dev`
- User: `langchain_user`
- Password: `langchain_password`

#### 14. Qdrant Vector Database
**Port**: `6333` (REST), `6334` (gRPC)
**Purpose**: Vector storage for embeddings

**Features**:
- High-performance vector search
- REST and gRPC APIs
- Persistent storage

---

## Deployment & Configuration

### 🚀 Startup Process

The system uses a phased startup approach:

1. **Phase 1**: Docker services (MindsDB)
2. **Phase 2**: GPU services (Medical + Face Recognition)
3. **Phase 3**: Core CPU services (Video + YOLO)
4. **Phase 4**: Specialized CPU services
5. **Phase 5**: Docker support (Ollama)
6. **Phase 6**: Database APIs (Vanna)
7. **Phase 7**: Web UI and dependent APIs

### 🔧 Environment Variables

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES="0"  # GPU device
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# API URLs
DEEPSEEK_API_URL="http://localhost:8000/analyze/"
OLLAMA_CHAT_URL="http://localhost:11434/api/chat"

# Database Configuration
POSTGRES_DB=langchain_dev
POSTGRES_USER=langchain_user
POSTGRES_PASSWORD=langchain_password

# S3 Configuration
AWS_S3_BUCKET=vision-app-prod-uploads

# TensorFlow Configuration
TF_FORCE_GPU_ALLOW_GROWTH="true"
TF_CPP_MIN_LOG_LEVEL="2"
```

### 📦 Virtual Environment Management

The system manages 10 separate virtual environments:

| Environment | Purpose | Requirements File |
|-------------|---------|-------------------|
| `venv_medical_gpu` | Medical imaging (GPU) | `requirements.txt` |
| `venv_face_cpu` | Face recognition | `requirements_face_recognition.txt` |
| `venv_video` | Video processing | `requirements_video.txt` |
| `venv_yolo_cpu` | YOLO detection | `requirements_yolo.txt` |
| `venv_metal_cpu` | Thermal gun detection | `requirements_thermal_gun_api.txt` |
| `venv_object_detection_cpu` | Object detection | `requirements__object_detection.txt` |
| `venv_translation_cpu` | Translation | `requirements_translation.txt` |
| `venv_vanna_api` | Vanna SQL | `requirements_vanna_api.txt` |
| `venv_robust_sql_api` | Robust SQL | `requirements_robust_sql_api.txt` |
| `venv_retrieval_cpu` | Embedding retrieval | `requirements_retrieval.txt` |

---

## API Reference

### 🏥 Medical Imaging API

#### POST /analyze/
Analyze medical images, extract OCR text, or detect objects.

**Request**:
```json
{
  "file": "image_file",
  "prompt": "Analyze this chest X-ray for abnormalities",
  "analysis_type": "medical"  // "medical", "ocr", "object_detection"
}
```

**Response**:
```json
{
  "response": "Analysis results...",
  "analysis_type": "medical",
  "filename": "chest_xray.jpg",
  "disclaimer": "This AI analysis is for educational purposes only..."
}
```

### 👤 Face Recognition API

#### POST /match_faces
Match faces in video against reference images.

**Request**:
```json
{
  "video_url": "https://example.com/video.mp4",
  "photos_bucket": "reference-images",
  "photos_prefix": "faces/",
  "high_confidence_threshold": 0.30,
  "medium_confidence_threshold": 0.40,
  "low_confidence_threshold": 0.50,
  "include_face_attributes": true,
  "include_anti_spoofing": true
}
```

**Response**:
```json
{
  "matches": [
    {
      "photo_key": "faces/person1.jpg",
      "frame_index": 5,
      "timestamp_ms": 5000,
      "confidence_score": 0.85,
      "confidence_level": "HIGH",
      "face_quality": {...},
      "face_attributes": {...}
    }
  ],
  "total_frames_processed": 30,
  "total_faces_detected": 15,
  "processing_time_seconds": 45.2
}
```

### 📹 Video API

#### POST /analyze_video
Analyze video content with AI.

**Request**:
```json
{
  "video_url": "https://example.com/video.mp4",
  "analysis_prompt": "Describe what's happening in this video"
}
```

**Response**:
```json
{
  "summary": "Video analysis summary...",
  "frame_analyses": [...],
  "processing_time": "45.2s",
  "frames_processed": 30
}
```

### 🔍 YOLO API

#### POST /detect_objects/
Detect objects in video using YOLOv8.

**Request**:
```json
{
  "video_url": "https://example.com/video.mp4"
}
```

**Response**:
```json
{
  "device": "cpu",
  "total_frames": 30,
  "total_detections": 150,
  "counts": {
    "person": 45,
    "car": 12,
    "dog": 3
  },
  "processing_time": "12.5s"
}
```

### 🗣️ Translation API

#### POST /translate
Translate text between languages.

**Request**:
```json
{
  "text": "Hello, how are you?",
  "target_language": "es",
  "source_language": "en"
}
```

**Response**:
```json
{
  "translated_text": "Hola, ¿cómo estás?",
  "source_language": "en",
  "target_language": "es",
  "generation_time_seconds": 1.2
}
```

### 🧠 Vanna Text-to-SQL API

#### POST /sql_query
Generate SQL from natural language.

**Request**:
```json
{
  "question": "How many users are there?"
}
```

**Response**:
```json
{
  "question": "How many users are there?",
  "sql_query": "SELECT COUNT(*) FROM users;",
  "result": [{"count": 1250}],
  "generation_time_seconds": 2.1
}
```

### 🤖 Robust Text-to-SQL API

#### POST /query
Enhanced SQL generation with MindsDB.

**Request**:
```json
{
  "question": "Show me all users who signed up last month"
}
```

**Response**:
```json
{
  "success": true,
  "sql": "SELECT * FROM users WHERE signup_date >= '2024-01-01'",
  "data": [...],
  "row_count": 45,
  "timing": {
    "generation_time": 1.5,
    "execution_time": 0.3,
    "total_time": 1.8
  }
}
```

### 🔍 Embedding & Retrieval API

#### POST /add
Add documents to vector store.

**Request**:
```json
{
  "collection_name": "documents",
  "texts": ["Document 1 content...", "Document 2 content..."]
}
```

**Response**:
```json
{
  "message": "Documents added successfully",
  "documents_received": 2,
  "chunks_added": 8
}
```

#### POST /search
Search for similar documents.

**Request**:
```json
{
  "collection_name": "documents",
  "query": "What is machine learning?",
  "k": 3
}
```

**Response**:
```json
{
  "matches": [
    {
      "page_content": "Machine learning is a subset of AI...",
      "metadata": {"source": "doc1.txt"}
    }
  ],
  "retrieval_time_seconds": 0.5
}
```

---

## Monitoring & Maintenance

### 📊 Health Monitoring

All services provide health check endpoints:

```bash
# Check all services
curl http://localhost:8000/health  # Medical Imaging
curl http://localhost:8001/health  # Video API
curl http://localhost:8002/health  # YOLO API
curl http://localhost:8004/health  # Face Recognition
curl http://localhost:8005/health  # Object Detection
curl http://localhost:8006/health  # Thermal Gun
curl http://localhost:8007/health  # Translation
curl http://localhost:8008/health  # Vanna API
curl http://localhost:8009/health  # Robust SQL
curl http://localhost:8010/health  # Embedding API
```

### 🖥️ System Monitoring Commands

```bash
# List all services
screen -list

# Monitor GPU usage
nvidia-smi

# Monitor RAM usage
free -h

# Check service logs
screen -r medical_imaging_gpu
screen -r face_recognition_gpu
screen -r video_api_cpu

# Check Docker containers
docker ps

# Monitor database connections
docker logs postgres-db
docker logs qdrant-db
docker logs ollama-service
```

### 🔧 Service Management

```bash
# Start all services
./start_all_services_optimized.sh

# Stop all services
./stop_all_services.sh

# Restart specific service
screen -S medical_imaging_gpu -X quit
screen -dmS medical_imaging_gpu bash -c "source venv_medical_gpu/bin/activate && uvicorn medical_imaging_api:app --host 0.0.0.0 --port 8000"

# Check service status
ps aux | grep uvicorn
ps aux | grep python
```

### 📈 Performance Monitoring

#### GPU Memory Monitoring
```python
def get_gpu_memory_info():
    """Get current GPU memory usage"""
    try:
        result = torch.cuda.memory_stats()
        allocated = result["allocated_bytes.all.current"] / 1024**2
        reserved = result["reserved_bytes.all.current"] / 1024**2
        free = torch.cuda.get_device_properties(0).total_memory / 1024**2 - reserved
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
            "free_percent": (free / (torch.cuda.get_device_properties(0).total_memory / 1024**2)) * 100
        }
    except Exception as e:
        logging.error(f"Failed to get GPU memory info: {e}")
        return None
```

#### Service Response Time Monitoring
```python
import time
import httpx

async def monitor_service_response_time(service_url: str):
    """Monitor service response time"""
    start_time = time.time()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{service_url}/health")
            response_time = time.time() - start_time
            return {
                "service": service_url,
                "status": response.status_code,
                "response_time": response_time,
                "healthy": response.status_code == 200
            }
    except Exception as e:
        return {
            "service": service_url,
            "status": "error",
            "response_time": time.time() - start_time,
            "error": str(e),
            "healthy": False
        }
```

---

## Troubleshooting

### 🚨 Common Issues

#### 1. GPU Memory Issues
**Symptoms**: CUDA out of memory errors
**Solutions**:
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size in medical imaging
MEMORY_CONFIG["max_image_size"] = 256  # Reduce from 320

# Restart GPU services
screen -S medical_imaging_gpu -X quit
screen -S face_recognition_gpu -X quit
# Restart services
```

#### 2. Service Connection Issues
**Symptoms**: Connection refused errors
**Solutions**:
```bash
# Check if ports are in use
lsof -i :8000
lsof -i :8001
# Kill processes if needed
kill -9 <PID>

# Restart specific service
screen -S service_name -X quit
# Restart via script
```

#### 3. Docker Container Issues
**Symptoms**: Container not starting
**Solutions**:
```bash
# Check container logs
docker logs mindsdb
docker logs ollama-service

# Restart containers
docker restart mindsdb
docker restart ollama-service

# Check container status
docker ps -a
```

#### 4. Virtual Environment Issues
**Symptoms**: Module not found errors
**Solutions**:
```bash
# Recreate virtual environment
rm -rf venv_service_name
python3 -m venv venv_service_name
source venv_service_name/bin/activate
pip install -r requirements_service_name.txt
```

### 🔍 Debugging Commands

```bash
# Check service logs
tail -f /var/log/syslog | grep uvicorn

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check memory usage
htop

# Monitor network connections
netstat -tulpn | grep :800

# Check disk space
df -h

# Monitor system resources
iostat -x 1
```

### 📋 Service Status Checklist

- [ ] All 14 services running
- [ ] GPU memory usage < 85%
- [ ] All health endpoints responding
- [ ] Docker containers healthy
- [ ] Database connections active
- [ ] Virtual environments activated
- [ ] Screen sessions active
- [ ] Ports not conflicting

### 🆘 Emergency Procedures

#### Complete System Restart
```bash
# Stop all services
./stop_all_services.sh

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Restart Docker containers
docker restart mindsdb ollama-service postgres-db qdrant-db

# Start all services
./start_all_services_optimized.sh
```

#### GPU Service Recovery
```bash
# Kill GPU services
screen -S medical_imaging_gpu -X quit
screen -S face_recognition_gpu -X quit

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Restart GPU services
screen -dmS medical_imaging_gpu bash -c "source venv_medical_gpu/bin/activate && uvicorn medical_imaging_api:app --host 0.0.0.0 --port 8000"
screen -dmS face_recognition_gpu bash -c "source venv_face_cpu/bin/activate && uvicorn face_recognition:app --host 0.0.0.0 --port 8004"
```

---

## Conclusion

The LC-DeepSeek-VL2 platform is a comprehensive AI/ML system with 14 specialized services covering computer vision, natural language processing, database operations, and web interfaces. The system is designed for production use with robust error handling, monitoring capabilities, and optimized resource allocation.

Key strengths:
- **Scalable Architecture**: Modular service design
- **Resource Optimization**: Intelligent GPU/CPU allocation
- **Comprehensive Coverage**: Full AI/ML pipeline
- **Production Ready**: Monitoring, logging, error handling
- **Flexible Deployment**: Docker and native service support

For support and maintenance, refer to the troubleshooting section and monitoring commands provided in this documentation.

---

**Documentation Version**: 1.0
**Last Updated**: 2024
**System Version**: LC-DeepSeek-VL2
**Model**: DeepSeek-VL2
