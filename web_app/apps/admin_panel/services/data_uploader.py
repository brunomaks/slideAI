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


class DataUploader:
    """Service for uploading and processing labeled training data."""
    
    def __init__(self):
        self.db_path = Path(settings.DATABASES['landmarks']['NAME'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_upload(self, uploaded_file: UploadedFile):
        """
        Handle ZIP file upload containing raw images.
        
        Args:
            uploaded_file: Django UploadedFile instance (ZIP)
            
        Returns:
            Dict with count of imported records
        """
        if not uploaded_file.name.lower().endswith('.zip'):
            raise ValueError("Only ZIP files are supported")
        
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
        version_id = f"v{timestamp}"
        
        try:
            resp = requests.post(
                f"{ml_api_url}/preprocess",
                json={
                    "dataset_version": version_id,
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
                    stats = self.get_upload_statistics()
                    return {'total': stats['total_samples']}
                    
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

    def get_upload_statistics(self):
        """
        Get statistics about uploaded data from SQLite.
        
        Returns:
            Dictionary with upload statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gestures_processed'")
                if not cursor.fetchone():
                    return {'total_samples': 0, 'by_label': []}
                
                # Count total
                cursor.execute("SELECT COUNT(*) FROM gestures_processed")
                total = cursor.fetchone()[0]
                
                # Count by label
                cursor.execute("SELECT gesture, COUNT(*) FROM gestures_processed GROUP BY gesture")
                by_label = [
                    {'label': row[0], 'count': row[1], 'dataset_type': 'all'} 
                    for row in cursor.fetchall()
                ]
                
                return {
                    'total_samples': total,
                    'by_label': by_label,
                }
        except Exception:
             return {'total_samples': 0, 'by_label': []}
