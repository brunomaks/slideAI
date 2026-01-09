# SlideAI

## Description

AI-powered web application that uses a standard webcam to interpret hand gestures in real time. A cloud-deployed machine learning pipeline processes the video stream directly in the user’s browser, detects hand positions, and classifies gestures. These recognized gestures are then mapped to slide navigation actions or other on-screen controls. The result is a fully contactless, accessible, and user-friendly way to interact with slideshows, no hardware required.


---

## Table of Contents

* [Prerequisites](#prerequisites)
* [Running the application with a pretrained model](#running-the-application-with-a-pretrained-model)
* [Dataset for model training](#dataset-for-model-training)
* [License](#license)

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

## Running the application with a pretrained model

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

* User Interface: [http://localhost:5173](http://localhost:5173)
* Admin Panel: [http://localhost:8001/admin/](http://localhost:8001/admin/)

---

## Dataset for model training
The source dataset we used for training can be found [here](https://www.kaggle.com/datasets/innominate817/hagrid-classification-512p).

We recommend using [this](https://chalmers-my.sharepoint.com/:f:/g/personal/maksymm_chalmers_se/IgA0dUolF7dFQ5A2M11-G8a3Adir32hl0w7KakaoqvU3cYc?e=PbbTAV) dataset for local training, as we have already trimmed it to only include the gestures supported by the application.
If you are having troubles accessing the link, try using incognito mode.


---

## License

This project is developed for academic purposes as part of DIT826 coursework at Chalmers University of Technology / University of Gothenburg.
