# ============ BASE STAGE ============
FROM tensorflow/tensorflow:2.16.1-gpu AS base

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ============ DEVELOPMENT STAGE ============
FROM base AS dev

# Development includes vim for debugging
RUN apt-get update && apt-get install -y vim && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8003
CMD ["python", "src/api.py"]

# ============ PRODUCTION STAGE ============
FROM base AS prod

COPY . .

EXPOSE 8003
# Production: optimized startup, no dev tools
CMD ["python", "src/api.py"]