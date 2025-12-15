"""Data upload service for handling labeled image uploads."""
import zipfile
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
    
    def handle_upload(self, uploaded_file: UploadedFile):
        """
        Handle ZIP file upload containing train/ and test/ folders.
        
        Args:
            uploaded_file: Django UploadedFile instance (ZIP)
            
        Returns:
            Dictionary with upload results
        """
        if not uploaded_file.name.endswith('.zip'):
            raise ValueError("Only ZIP files are supported")
        
        counts = {'train': 0, 'test': 0}
        
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
            
            # Process train and test datasets
            for dataset_type in ['train', 'test']:
                dataset_dir = temp_path / dataset_type
                if not dataset_dir.exists():
                    continue
                
                # Process each label folder
                for label_dir in dataset_dir.iterdir():
                    if not label_dir.is_dir():
                        continue
                    
                    label = label_dir.name
                    
                    # Process images in label folder
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
                                
                                counts[dataset_type] += 1
                            except Exception as e:
                                print(f"Error processing {image_file}: {e}")
                                continue
        
        return counts
    
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
