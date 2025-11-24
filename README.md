# Data-Intensive AI Application

## Software Engineering for Data-Intensive AI Applications

A production-ready machine learning application built with Django and TensorFlow, featuring automated model training, versioning, and web-based inference.

---

## 📋 Table of Contents

- [Architecture Overview](#️-architecture-overview)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Development Workflow](#-development-workflow)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Team](#-team-4)
- [License](#-license)

---

##  Architecture Overview

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

  - Note: You do NOT need to install the NVIDIA CUDA Toolkit on the host.

  - Linux: NVIDIA Container Toolkit (Required for Linux):

    The nvidia-docker wrapper is deprecated. Please install the NVIDIA Container Toolkit directly:

    ```bash
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

> **Note**: No Python installation required on host machine. Everything runs in Docker.

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

### 4. Start Web Application

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

```bash
# Apply configurations
kubectl apply -f kubernetes/persistent-volume.yaml
kubectl apply -f kubernetes/web-deployment.yaml
kubectl apply -f kubernetes/ml-training-job.yaml

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

## 🐛 Troubleshooting

### Common Issues

#### 1. GPU Not Detected

**Symptoms:**

```text
⚠️  CPU Mode: Running on CPU
```

**Solutions:**

- **Windows/WSL2**: Install NVIDIA drivers on Windows host (not in WSL)
- **Linux**: Install nvidia-docker2: `sudo apt install nvidia-docker2`
- **Verify**: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`

#### 2. Port 8000 Already in Use

**Solution:**

```bash
# Change port in docker-compose.yml
ports:
  - "8080:8000"  # Use port 8080 instead
```

#### 3. Permission Denied (Database/Models)

**Linux:**

```bash
# Fix ownership
sudo chown -R $USER:$USER shared_artifacts/
```

**Windows/macOS:**

```bash
# Remove and recreate
rm -rf shared_artifacts/data/*.sqlite
docker-compose run --rm web python manage.py migrate
```

#### 4. Container Fails to Build

**Clear Docker cache:**

```bash
docker-compose build --no-cache ml-training

docker-compose build --no-cache web
```

#### 5. Model Not Found Error

**Check active model:**

```bash
cat shared_artifacts/models/active_model.txt

# If missing, train a model:
docker-compose --profile training-cpu up
```

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

---

## 📚 Tech Stack Requirements

| Requirement | Implementation | Location |
|-------------|----------------|----------|
| **A. SQLite Database** | Shared volume with Django ORM | `shared_artifacts/data/` |
| **B. ML Pipeline** | TensorFlow training pipeline | `ml_service/src/train.py` |
| **C. Data Validation** | Schema validation + unit tests | `ml_service/src/data_validator.py` |
| **D. End-User Interface** | Django web UI for predictions | `web_app/apps/inference/` |
| **E. Model Versioning** | Version tracking and rollback | `web_app/apps/admin_panel/models.py` |
| **F. Admin Interface** | Dynamic retraining UI | `web_app/apps/admin_panel/` |
| **G. Docker/K8s Deployment** | Multi-container + K8s configs | `docker-compose.yml`, `kubernetes/` |

---

## 👥 Team 4

> To be updated

### Individual Contributions

See `DIT826-Individual_Contribution_Form.docx` for detailed breakdown.

---

## 📖 References

### Course Materials

- **Sculley et al.** - Hidden Technical Debt in Machine Learning Systems
- **Amershi et al.** - Software Engineering for Machine Learning: A Case Study
- **Breck et al.** - Data Validation for Machine Learning
- **Zinkevich** - Rules of Machine Learning
- **Hulten** - Building Intelligent Systems (Chapters 1-20)

### Technologies

- [Django 4.2 Documentation](https://docs.djangoproject.com/)
- [TensorFlow 2.15 Documentation](https://www.tensorflow.org/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

---

## 📄 License

This project is developed for academic purposes as part of DIT826 coursework at Chalmers University of Technology / University of Gothenburg.

**Academic Integrity Notice**: Code developed for course requirements. Please consult course policy before reuse.

---

## 🔄 Project Status

**Current Phase**: Active Development (Week 1 of 7)

### Completed

- ✅ Docker multi-container setup
- ✅ GPU/CPU training support
- ✅ Basic Django application structure
- ✅ Model versioning system

### In Progress

- 🔄 Data validation implementation
- 🔄 Admin panel UI
- 🔄 Model evaluation metrics

### Planned

- 📋 Kubernetes deployment testing
- 📋 CI/CD pipeline setup
- 📋 Final documentation

---

Built with ❤️ for DIT826 - Software Engineering for Data-Intensive AI Applications
