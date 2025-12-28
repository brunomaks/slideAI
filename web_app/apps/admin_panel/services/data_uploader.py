import json
import sqlite3
from pathlib import Path
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.db.models import Count
from PIL import Image
import tempfile


class DataUploader:
    """Service for uploading and processing labeled training data."""
    
    def __init__(self):
        self.db_path = Path(settings.DATABASES['landmarks']['NAME'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_upload(self, uploaded_file: UploadedFile):
        """
        Handle JSON file upload containing landmarks.
        
        Args:
            uploaded_file: Django UploadedFile instance (JSON)
            
        Returns:
            Dict with count of imported records
        """
        if not uploaded_file.name.lower().endswith('.json'):
            raise ValueError("Only JSON files are supported")
            
        try:
            # Parse JSON content
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON file")
            
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of records")
            
        # Write to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Re-create table
                cursor.execute("DROP TABLE IF EXISTS gestures_processed")
                cursor.execute("""
                    CREATE TABLE gestures_processed (
                        gesture TEXT,
                        landmarks TEXT
                    )
                """)
                
                count = 0
                for item in data:
                    gesture = item.get('gesture')
                    landmarks = item.get('landmarks')
                    
                    if gesture and landmarks:
                        cursor.execute(
                            "INSERT INTO gestures_processed (gesture, landmarks) VALUES (?, ?)", 
                            (gesture, json.dumps(landmarks))
                        )
                        count += 1
                
                conn.commit()
                
 
            
            return {'train': count, 'test': 0} # Tests are split at training time
            
        except Exception as e:
            raise RuntimeError(f"Failed to ingest landmarks: {e}")

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
