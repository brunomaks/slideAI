# Contributors:
# - Mahmoud
# - Maksym
# - Ahmet

# ============ BASE STAGE ============
FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============ DEVELOPMENT STAGE ============
FROM base AS dev

# Development: code will be mounted as volume, hot reload enabled
COPY . .
CMD ["python", "-m", "uvicorn", "config.asgi:application", "--host", "0.0.0.0", "--port", "8001", "--reload"]

# ============ PRODUCTION STAGE ============
FROM base AS prod

# Install production server with async support
RUN pip install gunicorn uvicorn

COPY . .

# Collect static files for production
RUN python manage.py collectstatic --noinput --settings=config.settings.production || true

# Production server: use uvicorn workers for async support
CMD ["gunicorn", "config.asgi:application", "--bind", "0.0.0.0:8001", "--workers", "4", "-k", "uvicorn.workers.UvicornWorker"]