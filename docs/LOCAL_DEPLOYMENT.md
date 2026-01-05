# Local Kubernetes Deployment (Docker Desktop)

This guide explains how to deploy the SlideAI application locally using Docker Desktop's built-in Kubernetes cluster. This is useful for development and testing without incurring cloud costs.

## Prerequisites

1.  **Docker Desktop** installed.
2.  **Kubernetes enabled** in Docker Desktop settings.
3.  `kubectl` installed and configured (usually handled by Docker Desktop).
    *   Verify: `kubectl config current-context` should be `docker-desktop`.

## 1. Prepare Docker Images

Since we are running locally, we don't need to push images to GitLab. However, Kubernetes expects the image names to match what is defined in the manifests.

Build and tag the images locally:

```bash
# Build/Tag Web App
docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/web:latest web_app/

# Build/Tag Frontend
docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/frontend:latest . -f frontend/Dockerfile

# Build/Tag ML Training (CPU version)
docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/ml-training:latest ml_service/

# Build/Tag Inference
docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/inference:latest inference/
```

> **Note**: The manifests use `imagePullPolicy: IfNotPresent`. By tagging them exactly as above, Kubernetes will verify they exist locally and skip trying to pull from the remote registry.

## 2. Configure Kubernetes

### Create Namespace
```bash
kubectl create namespace slideai
```

### Create Secrets
1.  **Registry Credentials** (Optional if using only local images, but good practice prevents "ImagePullBackOff" errors if it tries to pull):
    ```bash
    kubectl create secret docker-registry gitlab-registry \
      --docker-server=registry.git.chalmers.se \
      --docker-username=<your-username> \
      --docker-password=<your-token> \
      -n slideai
    ```

2.  **Application Secrets** (Django Key):
    The `kubernetes/03-secrets.yaml` file contains a default/dummy key. For local dev, this is fine.
    ```bash
    kubectl apply -f kubernetes/03-secrets.yaml
    ```

## 3. Deploy Application

Apply all Kubernetes manifests:

```bash
kubectl apply -f kubernetes/
```

This will create:
*   Storage (PersistentVolumeClaim)
*   ConfigMaps
*   Deployments (`web`, `frontend`, `ml-training`, `inference`)
*   Services

### Monitor Status
Watch the pods starting up:
```bash
kubectl get pods -n slideai -w
```
Wait until all pods are `Running`.

> **Troubleshooting**: If `web` or `inference` pods keep restarting (CrashLoopBackOff), it might be due to slow startup times on Docker Desktop causing "Liveness Probe" timeouts.
> **Fix**: Edit `kubernetes/06-web.yaml` and comment out the `livenessProbe` and `readinessProbe` sections, then re-apply.

## 4. Database Setup

The first time you deploy, the database is empty. You need to create an admin user manually.

1.  **Get the Web Pod name**:
    ```bash
    kubectl get pods -n slideai -l app=web
    ```
2.  **Run createsuperuser command**:
    Replace `<web-pod-name>` with the actual name (e.g., `web-746595c4c9-q2s8m`).
    ```bash
    kubectl exec -it -n slideai <web-pod-name> -- python /app/manage.py createsuperuser
    ```
    Follow the prompts to set username (e.g. `admin`) and password.

## 5. Access the Application

Docker Desktop exposes LoadBalancer services on `localhost`, but sometimes port mapping is tricky. The most reliable method is **Port Forwarding**.

### Access Admin Panel / API
Forward port 8001:
```bash
kubectl port-forward -n slideai service/web-service 8001:8001
```
*   Access: [http://localhost:8001/admin/](http://localhost:8001/admin/)

### Access Frontend
Forward port 8080 (mapping to internal port 80):
```bash
kubectl port-forward -n slideai service/frontend-service 8080:80
```
*   Access: [http://localhost:8080](http://localhost:8080)

## 6. Cleanup

To remove everything (including data):
```bash
kubectl delete namespace slideai
```
