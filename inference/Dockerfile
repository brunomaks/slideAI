# ============ BASE STAGE ============
FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============ DEVELOPMENT STAGE ============
FROM base AS dev

COPY . .
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]

# ============ PRODUCTION STAGE ============
FROM base AS prod

# Install production ASGI server
RUN pip install gunicorn

COPY . .

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8002", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120"]