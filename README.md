# Data-Intensive AI Applications

## Software Engineering for Data-Intensive AI Applications

A production-ready machine learning application built with Django and TensorFlow, featuring automated model training, versioning, and web-based inference.

---

## Table of Contents

* [Architecture Overview](#architecture-overview)
* [Prerequisites](#prerequisites)
* [Quick Start](#quick-start)
* [Development Workflow](#development-workflow)
* [Deployment](#deployment)
* [Debugging](#debugging)
* [License](#license)

---

## Architecture Overview

This project follows a **microservices architecture** with separate containers for ML training and web serving:

```text
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose                        │
├──────────────────────┬──────────────────────────────────┤
│   ML Training        │      Django Web App              │
│   Container          │      Container                   │
│                      │                                  │
│   • TensorFlow GPU   │   • REST API                     │
│   • Model Training   │   • User Interface               │
│   • Data Validation  │   • Admin Panel                  │
│   • Versioning       │   • Model Inference              │
└──────────┬───────────┴──────────────┬───────────────────┘
           │                          │
           └────────┬─────────────────┘
                    │
          ┌─────────▼─────────┐
          │  Shared Volumes   │
          │  • Models         │
          │  • Database       │
          └───────────────────┘
```

### Key Features

* **GPU/CPU Support**: Automatic hardware detection with fallback
* **Model Versioning**: Track and rollback models (Requirement E)
* **Data Validation**: Schema checks and quality verification (Requirement C)
* **Admin Interface**: Dynamic retraining and management (Requirement F)
* **REST API**: JSON endpoints for predictions (Requirement D)
* **Kubernetes Ready**: Production deployment configs included (Requirement G)

---

## Prerequisites

### Required

To use run the application locally, depending on your OS you are required to have:

* **Windows:** Docker Desktop with WSL2 backend
* **macOS:** Docker Desktop
* **Linux:** Docker Engine + Docker Compose

### Optional (For GPU Training)

* **NVIDIA GPU** with CUDA support
* **NVIDIA Drivers** installed on host (Docker handles CUDA/cuDNN internally).

> **Note**: You do NOT need to install the NVIDIA CUDA Toolkit on the host.
> **Note**: No Python installation required on host machine. Everything runs in Docker.

### GPU Configuration Guide

#### Windows 11

Create a `docker-compose.override.yml` file in the root directory:

```yaml
services:
  ml-training:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

#### Linux (Ubuntu & Fedora)

1. Verify driver installation:

```bash
nvidia-smi
```

2. Install NVIDIA Container Toolkit:

**Ubuntu:**

```bash
sudo apt-get install nvidia-container-toolkit
```

**Fedora:**

```bash
sudo dnf install nvidia-container-toolkit
```

3. Generate CDI specification:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

4. Restart Docker:

```bash
sudo systemctl restart docker
```

5. Verify setup:

```bash
docker run --rm -it --device=nvidia.com/gpu=all ubuntu nvidia-smi
```

6. Create Linux override:

```yaml
services:
  ml-training:
    devices:
      - nvidia.com/gpu=all
    volumes:
      - ./ml_service:/workspace:Z
      - ./shared_artifacts/models:/models:Z
      - ./shared_artifacts/data:/data:Z
      - ./shared_artifacts/images:/images:Z
```

---

## Quick Start (running the application with a pretrained model)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd team4
```

### 2. Initialize Database

```bash
docker compose run --rm web python manage.py migrate
docker compose run --rm web python manage.py createsuperuser
```

### 3. Start Application

**GPU:**

```bash
docker compose --profile gpu up
```

**CPU:**

```bash
docker compose --profile cpu up
```

### 4. Access the Application

* User Interface: [http://localhost:8000](http://localhost:8000)
* Admin Panel: [http://localhost:8000/admin-panel/](http://localhost:8000/admin-panel/)
* Django Admin: [http://localhost:8000/admin](http://localhost:8000/admin)

---

## Development Workflow

### Running Services Individually

```bash
docker compose --profile gpu up frontend
docker compose --profile gpu up web
docker compose --profile gpu up ml-training
docker compose --profile cpu up ml-training-cpu
```

### Common Tasks

Train model:

```bash
docker compose run --rm ml-training python src/train.py \
  --epochs 100 \
  --version v2 \
  --set-active
```

Django commands:

```bash
docker compose run --rm web python manage.py startapp new_app
docker compose run --rm web python manage.py makemigrations
docker compose run --rm web python manage.py shell
```

Run tests:

```bash
docker compose run --rm ml-training pytest
docker compose run --rm web pytest
```

### Live Development

* `ml_service/` mounted for live ML edits
* `web_app/` auto-reloads Django
* No rebuilds needed during development

### Adding Python Packages

Edit `requirements.txt` and rebuild:

```bash
docker compose build ml-training
docker compose build web
```

### Database Management

* Location: `shared_artifacts/data/database.sqlite`
* Reset:

```bash
rm shared_artifacts/data/database.sqlite
docker compose run --rm web python manage.py migrate
```

---

## Deployment

### Production Checklist

1. Set environment variables (`SECRET_KEY`, `ALLOWED_HOSTS`, `DEBUG=False`).
2. Build and push images:

```bash
docker build -t <your-repo>/web:latest ./web_app
docker build -t <your-repo>/ml-training:latest ./ml_service

docker push <your-repo>/web:latest
docker push <your-repo>/ml-training:latest
```

3. Deploy to Kubernetes:

```bash
kubectl apply -f kubernetes/01-storage.yaml
kubectl apply -f kubernetes/02-ml-deployment.yaml
kubectl apply -f kubernetes/03-web.yaml
kubectl apply -f kubernetes/04-frontend.yaml

kubectl get pods
kubectl get services
```

4. Trigger initial training:

```bash
kubectl create job --from=cronjob/ml-training manual-training-1
```

---

## Debugging

```bash
docker compose logs -f web
docker compose logs -f ml-training

docker compose exec web bash
docker compose exec ml-training bash
```

---

## License

This project is developed for academic purposes as part of DIT826 coursework at Chalmers University of Technology / University of Gothenburg.
