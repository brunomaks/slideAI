FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /workspace

RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose API port
EXPOSE 8003

# Default: run API server (can be overridden for training-only)
CMD ["python", "src/train.py"]