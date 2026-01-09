#!/usr/bin/env pwsh

# Contributors:
# - Ahmet
# - Mahmoud

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "stop", "start", "delete", "status", "push-images", "clean")]
    [string]$Action = "deploy",
    
    [string]$Project = "",
    [string]$Region = "europe-north1",
    [string]$Zone = "europe-north1-a",
    [string]$ClusterName = "slideai-cluster"
)

$ErrorActionPreference = "Stop"
$DEFAULT_PROJECT = "healthy-dolphin-481919-m9"
$PROJECT_ROOT = (Get-Item (Split-Path $MyInvocation.MyCommand.Path)).Parent.FullName
$GKE_DIR = Join-Path $PROJECT_ROOT "kubernetes\gke"
$NAMESPACE = "slideai"
$REGISTRY = "gcr.io"

# Example usage:
#   .\deploy-gke.ps1
#   .\deploy-gke.ps1 -Action push-images
#   .\deploy-gke.ps1 -Action clean

# Validate project is set
if (-not $Project) {
    $Project = $DEFAULT_PROJECT
}

$REGISTRY_PREFIX = "$REGISTRY/$Project"

function Write-Step($message) {
    Write-Host "`n=== $message ===" -ForegroundColor Cyan
}

function Deploy-Cluster {
    Write-Step "Creating GKE Cluster (cost-optimized for free trial)"
    
    # Check if cluster exists
    $existing = gcloud container clusters list --project $Project --zone $Zone --format="value(name)" 2>$null | Where-Object { $_ -eq $ClusterName }
    if ($existing) {
        Write-Host "Cluster '$ClusterName' already exists. Getting credentials..." -ForegroundColor Yellow
        gcloud container clusters get-credentials $ClusterName --zone $Zone --project $Project
    } else {
        # Create cost-optimized single-node cluster
        gcloud container clusters create $ClusterName `
            --project $Project `
            --zone $Zone `
            --machine-type "e2-standard-4" `
            --num-nodes 1 `
            --disk-size "30" `
            --disk-type "pd-standard" `
            --spot `
            --no-enable-autoupgrade `
            --no-enable-autorepair `
            --metadata disable-legacy-endpoints=true
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Failed to create cluster. Exiting." -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Cluster created successfully!" -ForegroundColor Green
    }
    
    # Enable required APIs
    Write-Step "Enabling required APIs"
    gcloud services enable container.googleapis.com --project $Project
    gcloud services enable containerregistry.googleapis.com --project $Project
    
    # Configure kubectl
    gcloud container clusters get-credentials $ClusterName --zone $Zone --project $Project
    
    # Wait for cluster API to be fully ready
    Write-Host "Waiting for cluster API to stabilize..."
    Start-Sleep -Seconds 10
    
    # Deploy NGINX Ingress Controller
    Write-Step "Deploying NGINX Ingress Controller"
    kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
    
    Write-Host "Waiting for ingress controller..." -ForegroundColor Yellow
    kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=300s
    
    # Install cert-manager for SSL
    Write-Host "Installing cert-manager..." -ForegroundColor Yellow
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml 2>$null
    Start-Sleep -Seconds 20
    
    # Push images if not already done
    Push-Images
    
    # Deploy application
    Deploy-Application
    
    # Get external IP
    Write-Host "`nWaiting for LoadBalancer IP..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    $externalIp = kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2>$null
    
    if ($externalIp) {
        Write-Host "Application ready at: http://$externalIp/" -ForegroundColor Green
        Write-Host "slideai.dedyn.io should point to this IP for proper SSL functionality." -ForegroundColor Green
    } else {
        Write-Host "IP not ready yet. Check: kubectl get svc -n ingress-nginx" -ForegroundColor Yellow
    }
}

function Push-Images {
    Write-Step "Building and pushing Docker images to GCR"
    
    # Configure docker for GCR
    gcloud auth configure-docker --quiet
    
    # Build and push each image
    $images = @(
        @{Name="frontend"; Path="$PROJECT_ROOT/frontend"; Target="prod"},
        @{Name="web"; Path="$PROJECT_ROOT/web_app"; Target="prod"},
        @{Name="ml-inference-landmarks"; Path="$PROJECT_ROOT/ml_inference_landmarks"; Target=$null},
        @{Name="ml-training-landmarks"; Path="$PROJECT_ROOT/ml_service_landmarks"; Target=$null}
    )
    
    foreach ($img in $images) {
        $tag = "$REGISTRY_PREFIX/$($img.Name):latest"
        Write-Host "Building $($img.Name)..." -ForegroundColor Yellow
        if ($img.Target) {
            docker build -t $tag $img.Path --target $($img.Target) --quiet
        } else {
            docker build -t $tag $img.Path --quiet
        }
        Write-Host "Pushing $($img.Name)..." -ForegroundColor Yellow
        docker push $tag
    }
    
    Write-Host "All images synced to GCR!" -ForegroundColor Green
}

function Deploy-Application {
    Write-Step "Deploying Kubernetes resources"
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Generate and apply secret
    Write-Host "Generating Django secret key..."
    $secretKey = python -c "import secrets; print(secrets.token_urlsafe(50))"
    kubectl create secret generic app-secrets -n $NAMESPACE --from-literal=DJANGO_SECRET_KEY="$secretKey" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image references in manifests to use GCR
    Write-Host "Applying manifests..."
    
    # Apply storage and config (includes NFS server)
    kubectl apply -f "$GKE_DIR/01-storage.yaml"
    kubectl apply -f "$GKE_DIR/02-configmaps.yaml"
    Start-Sleep -Seconds 15
    
    # Apply deployments with image substitution
    $deployments = @("04-ml-deployment.yaml", "05-inference-deployment.yaml", "06-web.yaml", "07-frontend.yaml")
    foreach ($file in $deployments) {
        $content = Get-Content "$GKE_DIR/$file" -Raw
        $content = $content -replace "registry\.git\.chalmers\.se/courses/dit826/2025/team4", "$REGISTRY_PREFIX"
        $content | kubectl apply -f -
    }
    
    # Wait for web pod to be ready before running migrations
    Write-Host "Waiting for web pod to be ready..."
    kubectl wait --for=condition=ready pod -l app=web -n $NAMESPACE --timeout=180s 2>$null
    
    # Run Django migrations
    Write-Host "Running Django migrations..." -ForegroundColor Yellow
    kubectl exec -n $NAMESPACE -c web deployment/web -- python /app/manage.py makemigrations core --settings=config.settings.development 2>$null
    kubectl exec -n $NAMESPACE -c web deployment/web -- python /app/manage.py migrate --settings=config.settings.development 2>$null
    Write-Host "Migrations complete!" -ForegroundColor Green
    
    # Copy local model files to shared volume using web pod
    Write-Host "Copying model files to shared volume..." -ForegroundColor Yellow
    Push-Location "$PROJECT_ROOT/shared_artifacts/models"
    $webPod = kubectl get pods -n $NAMESPACE -l app=web -o jsonpath="{.items[0].metadata.name}" 2>$null
    if ($webPod) {
        kubectl cp "active_model.json" "$NAMESPACE/${webPod}:/models/active_model.json" -c web 2>$null
        kubectl cp "gesture_model_123_20260103_163740.keras" "$NAMESPACE/${webPod}:/models/gesture_model_123_20260103_163740.keras" -c web 2>$null
        kubectl cp "hand_landmarker.task" "$NAMESPACE/${webPod}:/models/hand_landmarker.task" -c web 2>$null
        Write-Host "Model files copied to shared volume" -ForegroundColor Green
    } else {
        Write-Host "Warning: No web pod found for copying models" -ForegroundColor Yellow
    }
    Pop-Location
    
    # Wait for ml-training-landmarks and ml-inference-landmarks to become ready now that models are available
    Write-Host "Waiting for ML services to start..."
    kubectl rollout status deployment/ml-training-landmarks -n $NAMESPACE --timeout=300s 2>$null
    kubectl rollout status deployment/ml-inference-landmarks -n $NAMESPACE --timeout=300s 2>$null
    Write-Host "ML services ready!" -ForegroundColor Green
    
    # Apply ingress
    kubectl apply -f "$GKE_DIR/08-ingress.yaml"
    
    # Wait for pods
    Write-Host "Waiting for remaining pods..."
    kubectl wait --for=condition=ready pod -l app=frontend -n $NAMESPACE --timeout=180s 2>$null
    
    # Initialize admin
    kubectl exec -n $NAMESPACE -c web deployment/web -- python /app/initialize_admin.py 2>$null
    Write-Host "Deployment complete!" -ForegroundColor Green
}

function Stop-Cluster {
    Write-Host "Stopping cluster..." -ForegroundColor Cyan
    gcloud container clusters resize $ClusterName --node-pool default-pool --num-nodes 0 --zone $Zone --project $Project --quiet
    Write-Host "Cluster stopped (control plane still running)." -ForegroundColor Green
}

function Start-Cluster {
    Write-Host "Starting cluster..." -ForegroundColor Cyan
    gcloud container clusters resize $ClusterName --node-pool default-pool --num-nodes 1 --zone $Zone --project $Project --quiet
    gcloud container clusters get-credentials $ClusterName --zone $Zone --project $Project
    
    Write-Host "Waiting for cluster..."
    Start-Sleep -Seconds 30
    kubectl wait --for=condition=ready pod -l app=web -n $NAMESPACE --timeout=180s 2>$null
    kubectl exec -n $NAMESPACE -c web deployment/web -- python /app/initialize_admin.py 2>$null
    
    $externalIp = kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2>$null
    if ($externalIp) { Write-Host "Ready at: http://$externalIp/" -ForegroundColor Green }
}

function Delete-Cluster {
    Write-Host "WARNING: Delete cluster '$ClusterName'?" -ForegroundColor Red
    $confirm = Read-Host "Type 'yes' to confirm"
    
    if ($confirm -eq "yes") {
        gcloud container clusters delete $ClusterName --zone $Zone --project $Project --quiet
        Write-Host "Cluster deleted." -ForegroundColor Green
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
}

function Get-Status {
    Write-Host "Cluster Status:" -ForegroundColor Cyan
    gcloud container clusters describe $ClusterName --zone $Zone --project $Project --format="table(name,status,currentNodeCount)" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        kubectl get pods -n $NAMESPACE 2>$null
        $externalIp = kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath="{.status.loadBalancer.ingress[0].ip}" 2>$null
        if ($externalIp) { Write-Host "URL: http://$externalIp/" -ForegroundColor Green }
    }
}

function Clean-Resources {
    Write-Host "WARNING: Remove all configs, deployments, and volumes?" -ForegroundColor Red
    Write-Host "The cluster will remain. You can redeploy from scratch." -ForegroundColor Yellow
    $confirm = Read-Host "Type 'yes' to confirm"
    
    if ($confirm -eq "yes") {
        Write-Step "Deleting namespace and all resources"
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        
        Write-Host "Waiting for namespace deletion..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        
        Write-Step "Deleting storage resources"
        kubectl delete pvc --all -n $NAMESPACE --ignore-not-found 2>$null
        
        Write-Host "Clean complete! Cluster is ready for fresh deployment." -ForegroundColor Green
        Write-Host "Run: .\deploy-gke.ps1 -Project YOUR_PROJECT_ID" -ForegroundColor Cyan
    } else {
        Write-Host "Cancelled." -ForegroundColor Yellow
    }
}

# Main execution
switch ($Action) {
    "deploy" { Deploy-Cluster }
    "push-images" { Push-Images }
    "stop" { Stop-Cluster }
    "start" { Start-Cluster }
    "delete" { Delete-Cluster }
    "status" { Get-Status }
    "clean" { Clean-Resources }
}
