"""Data upload service for handling labeled image uploads."""
import io
import zipfile
from pathlib import Path
from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db import transaction
from django.db.models import Count
from PIL import Image
import tempfile

from apps.core.models import ImageMetadata


class DataUploader:
    """Service for uploading and processing labeled training data."""
    
    def __init__(self):
        # Use MEDIA_ROOT directly (points to /images which is shared_artifacts/images)
        self.base_path = Path(settings.MEDIA_ROOT)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def handle_upload(self, uploaded_file: UploadedFile):
        """
        Handle ZIP file upload containing train/ and test/ folders.

        This streams files from the ZIP to the destination without extracting
        the whole archive, and uses a transaction to keep DB writes consistent.
        Supports a leading root folder inside the ZIP.
        
        Args:
            uploaded_file: Django UploadedFile instance (ZIP)
        """

        if not uploaded_file.name.lower().endswith('.zip'):
            raise ValueError("Only ZIP files are supported")


        # Remove the entire images folder before uploading new dataset
        if self.base_path.exists() and self.base_path.is_dir():
            import shutil
            try:
                shutil.rmtree(self.base_path, ignore_errors=True)
            except Exception:
                # Fallback: remove files one by one if rmtree fails
                for child in self.base_path.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except Exception:
                            pass
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Clear the ImageMetadata table so only new images are shown
        ImageMetadata.objects.all().delete()

        counts = {'train': 0, 'test': 0}
        allowed_exts = {'.jpg', '.jpeg', '.png'}

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=True) as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp.flush()

            if not zipfile.is_zipfile(tmp.name):
                raise ValueError("Invalid ZIP file")

            with zipfile.ZipFile(tmp.name, 'r') as zf, transaction.atomic():
                for member in zf.namelist():
                    if member.endswith('/'):
                        continue

                    path_parts = Path(member).parts
                    if 'train' in path_parts:
                        dataset_type = 'train'
                    elif 'test' in path_parts:
                        dataset_type = 'test'
                    else:
                        continue  # skip anything outside train/test

                    # Expect at least dataset_type/label/file
                    try:
                        idx = path_parts.index(dataset_type)
                        label = path_parts[idx + 1]
                    except (ValueError, IndexError):
                        continue

                    filename = Path(member).name
                    ext = Path(filename).suffix.lower()
                    if ext not in allowed_exts:
                        continue

                    try:
                        with zf.open(member) as source:
                            data = source.read()

                        # Verify image
                        with Image.open(io.BytesIO(data)) as img:
                            width, height = img.size

                        dest_dir = self.base_path / dataset_type / label
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / filename
                        with open(dest_path, 'wb') as out_file:
                            out_file.write(data)

                        ImageMetadata.objects.create(
                            filename=filename,
                            label=label,
                            width=width,
                            height=height,
                            file_path=str(dest_path),
                            dataset_type=dataset_type,
                            source_dataset='user_upload'
                        )

                        counts[dataset_type] += 1
                    except Exception as exc:  # skip bad files but keep going
                        print(f"Error processing {member}: {exc}")
                        continue

        if counts['train'] == 0 and counts['test'] == 0:
            raise ValueError("No train/test images found in ZIP. Expected train/ and test/ folders with label subfolders.")

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
