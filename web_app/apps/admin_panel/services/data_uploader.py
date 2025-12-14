"""Data upload service for handling labeled image uploads."""
import zipfile
import csv
from pathlib import Path
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db.models import Count
from PIL import Image
import shutil
import tempfile

from apps.core.models import ImageMetadata


class DataUploader:
    """Service for uploading and processing labeled training data."""
    
    def __init__(self):
        self.base_path = Path(settings.MEDIA_ROOT) / 'uploaded_data'
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def handle_upload(self, uploaded_file: UploadedFile, dataset_type: str, label: str = None):
        """
        Handle file upload and process data.
        
        Args:
            uploaded_file: Django UploadedFile instance
            dataset_type: 'train', 'test', or 'validation'
            label: Optional gesture label for single-class uploads
            
        Returns:
            Dictionary with upload results
        """
        if uploaded_file.name.endswith('.zip'):
            return self._handle_zip_upload(uploaded_file, dataset_type)
        elif uploaded_file.name.endswith('.csv'):
            return self._handle_csv_upload(uploaded_file, dataset_type)
        else:
            raise ValueError("Unsupported file format. Please upload ZIP or CSV.")
    
    def _handle_zip_upload(self, uploaded_file: UploadedFile, dataset_type: str):
        """
        Handle ZIP file upload containing images in labeled folders.
 
        Args:
            uploaded_file: ZIP file
            dataset_type: Dataset type
            
        Returns:
            Dictionary with count of processed images
        """
        count = 0
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / uploaded_file.name
            
            # Save uploaded file
            with open(zip_path, 'wb+') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_path)
            
            # Process images
            for label_dir in temp_path.iterdir():
                if label_dir.is_dir() and label_dir.name != uploaded_file.name:
                    label = label_dir.name
                    
                    for image_file in label_dir.glob('*'):
                        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            try:
                                # Verify image can be opened
                                with Image.open(image_file) as img:
                                    width, height = img.size
                                
                                # Copy to media directory
                                dest_dir = self.base_path / dataset_type / label
                                dest_dir.mkdir(parents=True, exist_ok=True)
                                dest_path = dest_dir / image_file.name
                                shutil.copy2(image_file, dest_path)
                                
                                # Create database record
                                ImageMetadata.objects.create(
                                    filename=image_file.name,
                                    label=label,
                                    width=width,
                                    height=height,
                                    file_path=str(dest_path),
                                    dataset_type=dataset_type,
                                    source_dataset='user_upload'
                                )
                                
                                count += 1
                            except Exception as e:
                                print(f"Error processing {image_file}: {e}")
                                continue
        
        return {
            'count': count,
            'dataset_type': dataset_type,
        }
    
    def _handle_csv_upload(self, uploaded_file: UploadedFile, dataset_type: str):
        """
        Handle CSV file upload with image metadata.
        
        Expected CSV format:
        filename,label,file_path,width,height
        image1.jpg,like,/path/to/image1.jpg,128,128
        
        Args:
            uploaded_file: CSV file
            dataset_type: Dataset type
            
        Returns:
            Dictionary with count of processed records
        """
        count = 0
        
        # Decode CSV
        decoded_file = uploaded_file.read().decode('utf-8').splitlines()
        reader = csv.DictReader(decoded_file)
        
        for row in reader:
            try:
                # Verify required fields
                if not all(k in row for k in ['filename', 'label', 'file_path']):
                    continue
                
                # Get dimensions if not provided
                width = int(row.get('width', 0))
                height = int(row.get('height', 0))
                
                if width == 0 or height == 0:
                    # Try to read from file
                    try:
                        with Image.open(row['file_path']) as img:
                            width, height = img.size
                    except:
                        width, height = 128, 128  # Default
                
                # Create database record
                ImageMetadata.objects.create(
                    filename=row['filename'],
                    label=row['label'],
                    width=width,
                    height=height,
                    file_path=row['file_path'],
                    dataset_type=dataset_type,
                    source_dataset='user_upload'
                )
                
                count += 1
            except Exception as e:
                print(f"Error processing row {row}: {e}")
                continue
        
        return {
            'count': count,
            'dataset_type': dataset_type,
        }
    
    def get_upload_statistics(self):
        """
        Get statistics about uploaded data.
        
        Returns:
            Dictionary with upload statistics
        """
        total_images = ImageMetadata.objects.filter(source_dataset='user_upload').count()
        
        by_label = ImageMetadata.objects.filter(
            source_dataset='user_upload'
        ).values('label', 'dataset_type').annotate(count=Count('id'))
        
        return {
            'total_images': total_images,
            'by_label': list(by_label),
        }
