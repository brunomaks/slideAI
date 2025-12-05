# Data-Intensive AI Applications

## Software Engineering for Data-Intensive AI Applications

A production-ready machine learning application built with Django and TensorFlow, featuring automated model training, versioning, and web-based inference.

---

## 📋 Table of Contents

- [Architecture Overview](#️-architecture-overview)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Development Workflow](#-development-workflow)
- [Deployment](#-deployment)
- [Team](#-team-4)
- [License](#-license)

---

## 🏗️ Architecture Overview

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

- ✅ **GPU/CPU Support**: Automatic hardware detection with fallback
- ✅ **Model Versioning**: Track and rollback models (Requirement E)
- ✅ **Data Validation**: Schema checks and quality verification (Requirement C)
- ✅ **Admin Interface**: Dynamic retraining and management (Requirement F)
- ✅ **REST API**: JSON endpoints for predictions (Requirement D)
- ✅ **Kubernetes Ready**: Production deployment configs included (Requirement G)

---

## 📦 Prerequisites

### Required

- **Docker Desktop** (latest version)
  - Windows: Docker Desktop with WSL2 backend
  - macOS: Docker Desktop
  - Linux: Docker Engine + Docker Compose

### Optional (For GPU Training)

- **NVIDIA GPU** with CUDA support
- **NVIDIA Drivers** installed on host (Docker handles CUDA/cuDNN internally).

> **Note**: You do NOT need to install the NVIDIA CUDA Toolkit on the host.

> **Note**: No Python installation required on host machine. Everything runs in Docker.

**Windows 11**:

To enable GPU support on Windows, make sure that you have docker-compose.override.yml file created and filled with:

```yaml
services:
  ml-training:
    deploy:
      resources:
        reservations:
          devices:
            driver: nvidia
            count: all
            capabilities: [gpu] 
```

**Linux**:

Make sure you have the nvidia driver installed by running

```bash
nvidia-smi
``` 

Install NVIDIA Container Toolkit

Ubuntu:

```bash
sudo apt-get install nvidia-container-toolkit
```

Fedora:

```bash
sudo dnf install nvidia-container-toolkit
```

Generate CDI specification and verify it

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

Restart the docker service

```bash
sudo systemctl restart docker
```

Verify the setup by running nvidia-smi inside the container

```bash
docker run --rm -it --device=nvidia.com/gpu=all ubuntu nvidia-smi
```

To enable GPU access and proper SELinux labeling for the ML training service, create a `docker-compose.override.yml` file with the following content:

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
### What this does:
- **GPU Access**: Enables all available NVIDIA GPUs for the `ml-training` service using CDI specification
- **SELinux Labels**: The `:Z` flag applies proper SELinux context labels to ensure the mounted volumes are accessible only from their respective containers

> **Note**: The `:Z` flag modifies SELinux labels on your host system. Use with caution, especially when mounting system directories.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd team4
```

### 2. Initialize Database

```bash
docker-compose run --rm web python manage.py migrate
docker-compose run --rm web python manage.py createsuperuser
```

### 3. Train Initial Model

**For GPU (Windows/Linux with NVIDIA GPU):**

```bash
docker-compose --profile training up
```

**For CPU (macOS or systems without GPU):**

```bash
docker-compose --profile training-cpu up
```

### 4. Start Web + Frontend Application

```bash
docker-compose up
```

or

```bash
docker-compose up web
```

### 5. Access the Application

- **User Interface**: <http://localhost:8000>
- **Admin Panel**: <http://localhost:8000/admin-panel/>
- **Django Admin**: <http://localhost:8000/admin>

---

## 💻 Development Workflow

### Running Services Individually

**Run specific services directly:**

```bash
# Frontend & Web services only
docker-compose up frontend

# Web service only
docker-compose up web

# GPU training only
docker-compose up ml-training

# CPU training only
docker-compose up ml-training-cpu
```

**Train model with custom parameters:**

```bash
docker-compose run --rm ml-training python src/train.py \
  --epochs 100 \
  --version v2 \
  --set-active
```

**Run Django management commands:**

```bash
# Create new Django app
docker-compose run --rm web python manage.py startapp new_app

# Create database migrations
docker-compose run --rm web python manage.py makemigrations

# Open Django shell
docker-compose run --rm web python manage.py shell
```

**Run tests:**

```bash
# ML service tests
docker-compose run --rm ml-training pytest

# Django tests
docker-compose run --rm web pytest
```

### Live Development

Both containers mount source code as volumes for **live code editing**:

- Edit files in `ml_service/` → Changes reflect immediately
- Edit files in `web_app/` → Django auto-reloads
- No container rebuild needed during development

### Adding Python Packages

**For ML service:**

```bash
# Edit ml_service/requirements.txt
# Then rebuild container:
docker-compose build ml-training
```

**For web app:**

```bash
# Edit web_app/requirements.txt
# Then rebuild container:
docker-compose build web
```

### Database Management

**View database:**

```bash
# Access SQLite CLI
docker-compose run --rm web python manage.py dbshell

# Or use SQLite browser on host:
sqlite3 shared_artifacts/data/database.sqlite
```

**Reset database:**

```bash
# Remove database file
rm shared_artifacts/data/database.sqlite

# Recreate
docker-compose run --rm web python manage.py migrate
```

---

## 🚢 Deployment

### Production Checklist

**1. Update settings:**

```bash
# Set environment variables
export SECRET_KEY='your-production-secret-key'
export ALLOWED_HOSTS='yourdomain.com,www.yourdomain.com'
export DEBUG='False'
```

**2. Build production images:**

```bash
docker build -t gcr.io/YOUR_PROJECT/web:latest ./web_app
docker build -t gcr.io/YOUR_PROJECT/ml-training:latest ./ml_service

docker push gcr.io/YOUR_PROJECT/web:latest
docker push gcr.io/YOUR_PROJECT/ml-training:latest
```

**3. Deploy to Kubernetes:**

Build and push all 3 images.

Replace <your-username>/... in the files with your actual image names.

Run:

```bash
# Apply configurations
kubectl apply -f kubernetes/01-storage.yaml
kubectl apply -f kubernetes/02-ml-deployment.yaml
kubectl apply -f kubernetes/03-web.yaml
kubectl apply -f kubernetes/04-frontend.yaml

# Check status
kubectl get pods
kubectl get services
```

**4. Trigger model training:**

```bash
# Create training job
kubectl create job --from=cronjob/ml-training manual-training-1
```

### Cloud Platform Guides

**Google Kubernetes Engine (GKE):**

```bash
# Create cluster
gcloud container clusters create team4-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2

# Deploy
kubectl apply -f kubernetes/
```

**AWS EKS / Azure AKS:**
See `kubernetes/README.md` for platform-specific instructions.

---

### Debug Mode

**View container logs:**

```bash
# Web service logs
docker-compose logs -f web

# Training logs
docker-compose logs -f ml-training
```

**Enter running container:**

```bash
docker-compose exec web bash
docker-compose exec ml-training bash
```

## 👥 Team 4

> To be updated

## 📖 References

> To be updated

### Technologies

- [Django Documentation](https://docs.djangoproject.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

## 📄 License

This project is developed for academic purposes as part of DIT826 coursework at Chalmers University of Technology / University of Gothenburg.