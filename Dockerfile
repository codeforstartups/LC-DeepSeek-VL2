FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies (this layer will be cached unless requirements change)
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu117 -e . && \
    pip install fastapi uvicorn[standard]

# Now copy the rest of your code (this step will only invalidate the cache if your code changes)
COPY . /app

EXPOSE 80
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "80"]