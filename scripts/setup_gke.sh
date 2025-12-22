#!/bin/bash

# Exit on error
set -e

echo "Setting up GKE Cluster for slideAi..."

# Configuration
export CLUSTER_NAME="slideai-cluster"
export REGION="europe-west1"
export ZONE="$REGION-b"

# 1. Create Main Cluster
echo "Creating main cluster (sys/web nodes)..."
gcloud container clusters create $CLUSTER_NAME \
    --region $REGION \
    --num-nodes 1 \
    --machine-type e2-standard-2

# 2. Get Credentials
echo "Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# 3. Create GPU Node Pool
echo "Creating GPU node pool for ML training..."
gcloud container node-pools create gpu-pool \
    --cluster $CLUSTER_NAME \
    --region $REGION \
    --machine-type n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes 1 \
    --enable-autoscaling --min-nodes 0 --max-nodes 1

# 4. Install NVIDIA Drivers
echo "Installing NVIDIA drivers..."
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

echo "Cluster setup complete!"
echo "Next steps:"
echo "1. Create registry secret:"
echo "   kubectl create secret docker-registry gitlab-registry ..."
echo "2. Deploy app:"
echo "   kubectl apply -f kubernetes/"
