#!/usr/bin/env pwsh
# Minikube Deployment Script for SlideAI Project

param(
    [switch]$SkipBuild,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Get-Location
$NAMESPACE = "slideai"

if ($Clean) {
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    minikube delete
    Write-Host "Cleanup complete"
    exit 0
}

Write-Host "Starting Minikube..."
minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=20g

minikube addons enable ingress
minikube addons enable storage-provisioner
minikube addons enable default-storageclass

& minikube -p minikube docker-env --shell powershell | Invoke-Expression

if (-not $SkipBuild) {
    Write-Host "Building Docker images..."
    docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/frontend:latest "$PROJECT_ROOT/frontend"
    docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/web:latest "$PROJECT_ROOT/web_app"
    docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/inference:latest "$PROJECT_ROOT/inference"
    docker build -t registry.git.chalmers.se/courses/dit826/2025/team4/ml-training:latest "$PROJECT_ROOT/ml_service"
}

Write-Host "Deploying Kubernetes resources..."

kubectl apply -f "$PROJECT_ROOT/kubernetes/01-storage.yaml"
kubectl apply -f "$PROJECT_ROOT/kubernetes/02-configmaps.yaml"

Write-Host "Generating Django secret key..."
$secretKey = python -c "import secrets; print(secrets.token_urlsafe(50))"
kubectl create secret generic app-secrets -n $NAMESPACE --from-literal=DJANGO_SECRET_KEY="$secretKey" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f "$PROJECT_ROOT/kubernetes/04-ml-deployment.yaml"

Write-Host "Waiting for ml-training pod to be ready..."
kubectl wait --for=condition=ready pod -l app=ml-training -n $NAMESPACE --timeout=300s

Write-Host "Copying model files to shared volume..."
$mlPod = kubectl get pods -n $NAMESPACE -l app=ml-training -o jsonpath='{.items[0].metadata.name}'
kubectl cp "shared_artifacts\models\active_model.txt" "${mlPod}:/models/active_model.txt" -n $NAMESPACE
kubectl cp "shared_artifacts\models\gesture_model_20251125_193254.keras" "${mlPod}:/models/gesture_model_20251125_193254.keras" -n $NAMESPACE
kubectl cp "shared_artifacts\models\hand_landmarker.task" "${mlPod}:/models/hand_landmarker.task" -n $NAMESPACE

kubectl apply -f "$PROJECT_ROOT/kubernetes/05-inference-deployment.yaml"
kubectl apply -f "$PROJECT_ROOT/kubernetes/06-web.yaml"
kubectl apply -f "$PROJECT_ROOT/kubernetes/07-frontend.yaml"
kubectl apply -f "$PROJECT_ROOT/kubernetes/08-ingress.yaml"

Write-Host "Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=300s 2>$null
kubectl wait --for=condition=ready pod -l app=web -n $NAMESPACE --timeout=300s 2>$null
kubectl wait --for=condition=ready pod -l app=ml-training -n $NAMESPACE --timeout=300s 2>$null

Write-Host "`nChecking inference pods (may take longer)..."
kubectl wait --for=condition=ready pod -l app=inference -n $NAMESPACE --timeout=60s 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Inference pods not ready yet. Check with: kubectl get pods -n slideai"
}

kubectl get pods -n $NAMESPACE

Write-Host "`nDeployment Complete!"
Write-Host "`nSetting up port forwarding to ingress..."

Get-Job | Where-Object { $_.Name -like "port-forward-*" } | Remove-Job -Force

Start-Job -Name "port-forward-ingress" -ScriptBlock {
    kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 8080:80
} | Out-Null

Start-Sleep -Seconds 2

Write-Host "`nApplication is now accessible at:"
Write-Host "  http://localhost:8080/"
Write-Host "  http://localhost:8080/api/"
Write-Host "  http://localhost:8080/inference/"
Write-Host "`nNote: Port forwarding runs in background."
Write-Host "To stop: Get-Job -Name port-forward-ingress | Stop-Job | Remove-Job"
