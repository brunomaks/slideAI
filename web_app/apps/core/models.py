from django.db import models

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ImageMetadata(BaseModel):
    """Store metadata for training images."""
    filename = models.CharField(max_length=255)
    label = models.CharField(max_length=50, db_index=True)
    width = models.IntegerField()
    height = models.IntegerField()
    file_path = models.CharField(max_length=512)
    dataset_type = models.CharField(
        max_length=10, 
        choices=[('train', 'Training'), ('test', 'Test'), ('validation', 'Validation')],
        default='train',
        db_index=True
    )
    is_synthetic = models.BooleanField(default=False)
    source_dataset = models.CharField(max_length=100, default='hagrid')

    class Meta:
        db_table = 'image_metadata'
        indexes = [
            models.Index(fields=['label', 'dataset_type']),
        ]

    def __str__(self):
        return f"{self.filename} ({self.label})"

