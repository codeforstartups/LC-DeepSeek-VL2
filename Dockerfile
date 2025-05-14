FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install fastapi uvicorn[standard] && \
    pip install -e .

EXPOSE 80
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "80"]