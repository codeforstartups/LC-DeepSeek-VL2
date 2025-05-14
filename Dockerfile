# Use NVIDIA’s CUDA runtime for GPU support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repository code and install Python dependencies
COPY . /app
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip3 install -e .

# Copy and expose the FastAPI application
COPY inference_api.py /app/inference_api.py
EXPOSE 80
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "80"]
