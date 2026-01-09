# Contributors:
# - Mahmoud
# - Yaroslav

import json
import sqlite3
import requests
import time
import os
from pathlib import Path
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.db.models import Count
from apps.core.models import Dataset


class DataUploader:
    """Service for uploading and processing labeled training data."""
    def handle_upload(self, uploaded_file: UploadedFile, dataset_version: str, user: None):
        """
        Handle ZIP file upload containing raw images.
        
        Args:
            uploaded_file: Django UploadedFile instance (ZIP)
            
        Returns:
            Dict with count of imported records
        """
        if not uploaded_file.name.lower().endswith('.zip'):
            raise ValueError("Only ZIP files are supported")
        
        if Dataset.objects.filter(version=dataset_version).exists():
            raise ValueError(f"Dataset version '{dataset_version}' already exists")
        
        # 1. Save ZIP locally
        # Use a timestamped filename to avoid collisions
        timestamp = int(time.time())
        zip_filename = f"upload_{timestamp}.zip"
        fs_path = Path(settings.MEDIA_ROOT) / zip_filename
        fs_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fs_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
                
        # 2. Trigger Preprocessing via API
        ml_api_url = settings.ML_TRAINING_API_URL
        
        try:
            resp = requests.post(
                f"{ml_api_url}/preprocess",
                json={
                    "dataset_version": dataset_version,
                    "zip_filename": zip_filename
                },
                timeout=10
            )
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data['job_id']
            
            # Dynamic timeout based on file size
            # Estimate: ~35 samples/second processing rate + buffer
            # File size in MB * 100 gives rough sample estimate
            file_size_mb = uploaded_file.size / (1024 * 1024)
            estimated_samples = file_size_mb * 100  # rough estimate
            estimated_time = max(60, min(600, int(estimated_samples / 35) + 30))  # 60s min, 600s max
            
            # Poll for completion with dynamic timeout
            for _ in range(estimated_time):
                time.sleep(1)
                status_resp = requests.get(f"{ml_api_url}/preprocess/{job_id}", timeout=5)
                status_data = status_resp.json()
                
                if status_data['status'] == 'completed':
                    # Return stats
                    stats = status_data['message']
                    dataset = Dataset.objects.create(
                        version=dataset_version,
                        uploaded_by=user,
                        raw_samples=stats['total_raw_samples'],
                        raw_preprocessed_samples=stats['total_preprocessed_samples'],
                        validated_preprocessed_samples=stats['valid_preprocessed_samples'],
                        zip_filename=zip_filename,
                        label_stats=stats["label_stats"]
                    )

                    return {'total': stats['total_raw_samples'], 'dataset': dataset}
                    
                if status_data['status'] == 'failed':
                    raise RuntimeError(f"Preprocessing failed: {status_data.get('message')}")
                    
            raise RuntimeError("Preprocessing timed out")
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to communicate with ML service: {e}")
        finally:
            # Optional: Delete zip after successful handoff if desired, 
            # but user might want to keep it as backup. 
            # Current plan says "keep zip", so we leave it.
            pass