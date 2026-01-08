# Contributors:
# - Mahmoud

from django.db import models
from django.contrib.auth.models import User


class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Dataset(BaseModel):
    version = models.CharField(max_length=50, unique=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    raw_samples = models.IntegerField()
    raw_preprocessed_samples = models.IntegerField()
    validated_preprocessed_samples = models.IntegerField()
    zip_filename = models.CharField(max_length=255)
    label_stats = models.JSONField()

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.version} ({self.validated_preprocessed_samples} samples)"
    
    @classmethod
    def get_latest_statistics(cls):
        """Get statistics from the most recent dataset."""
        try:
            dataset = cls.objects.first()
            return {
                'label_stats': dataset.label_stats,
                'total_samples': dataset.validated_preprocessed_samples,
            }
        except:
            return {
                'label_stats': [],
                'total_samples': 0,
            }
    
    @classmethod
    def get_statistics_for_version(cls, version):
        """Get statistics for a specific dataset version."""
        try:
            dataset = cls.objects.get(version=version)
            return {
                'label_stats': dataset.label_stats,
                'total_samples': dataset.validated_preprocessed_samples,
            }
        except:
            return {
                'label_stats': [],
                'total_samples': 0,
            }

class ModelVersion(BaseModel):
    """Track trained model versions and their metadata."""
    version_id = models.CharField(max_length=100, unique=True, db_index=True)
    model_file = models.CharField(max_length=255)
    framework = models.CharField(max_length=50, default='tensorflow')

    # Training metadata
    training_date = models.DateTimeField()
    trained_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    training_duration_seconds = models.IntegerField(null=True, blank=True)

    # Dataset info
    class_labels = models.JSONField(null=True, blank=True)
    train_dataset_size = models.IntegerField()
    test_dataset_size = models.IntegerField(null=True, blank=True)
    validation_dataset_size = models.IntegerField(null=True, blank=True)

    # Hyperparameters
    epochs = models.IntegerField()
    batch_size = models.IntegerField()
    learning_rate = models.FloatField(null=True, blank=True)

    # Performance metrics
    train_accuracy = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField()
    train_loss = models.FloatField(null=True, blank=True)
    validation_loss = models.FloatField(null=True, blank=True)
    test_loss = models.FloatField(null=True, blank=True)

    # Evaluation metrics
    test_precision = models.FloatField(null=True, blank=True)
    test_recall = models.FloatField(null=True, blank=True)
    test_f1_score = models.FloatField(null=True, blank=True)
    confusion_matrix = models.JSONField(null=True, blank=True)

    # Deployment status
    is_active = models.BooleanField(default=False, db_index=True)
    deployment_date = models.DateTimeField(null=True, blank=True)
    deployed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='deployed_models'
    )

    # Notes
    description = models.TextField(blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        db_table = 'model_versions'
        ordering = ['-created_at']

    def __str__(self):
        status = "ACTIVE" if self.is_active else "INACTIVE"
        return f"{self.version_id} ({status}) - Acc: {self.test_accuracy:.2%}"


class Prediction(BaseModel):
    """Log all predictions made by the system."""
    request_id = models.CharField(max_length=100, db_index=True, blank=True)

    # Model info
    model_version = models.ForeignKey(
        ModelVersion,
        on_delete=models.SET_NULL,
        null=True,
        related_name='predictions'
    )

    # Prediction details
    predicted_class = models.CharField(max_length=50)
    confidence = models.FloatField()
    direction = models.CharField(max_length=5, null=True, blank=True)

    # Landmarks details
    handedness = models.CharField(max_length=5, null=True, blank=True)
    landmarks = models.JSONField(null=True, blank=True)

    # Performance tracking
    inference_time_ms = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['request_id', 'created_at']),
            models.Index(fields=['model_version', 'created_at']),
        ]

    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.1%}) at {self.created_at}"


class TrainingRun(BaseModel):
    """Track model training runs and their outcomes."""
    run_id = models.CharField(max_length=100, unique=True)
    model_version = models.OneToOneField(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name='training_run',
        null=True,
        blank=True
    )

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('cancelled', 'Cancelled')
        ],
        default='pending',
        db_index=True
    )

    # Training details
    started_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Configuration
    config = models.JSONField(default=dict)  # Store all hyperparameters

    # Results
    final_metrics = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    logs = models.TextField(blank=True)

    class Meta:
        db_table = 'training_runs'
        ordering = ['-created_at']

    def __str__(self):
        return f"Training Run {self.run_id} - {self.status}"


class UploadTask(BaseModel):
    """Track progress of dataset uploads."""
    task_id = models.CharField(max_length=100, unique=True, db_index=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    filename = models.CharField(max_length=255)

    # Progress tracking
    status = models.CharField(
        max_length=20,
        choices=[
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='processing',
        db_index=True
    )
    total_files = models.IntegerField(default=0)
    processed_files = models.IntegerField(default=0)
    train_count = models.IntegerField(default=0)
    test_count = models.IntegerField(default=0)

    # Results
    error_message = models.TextField(blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'upload_tasks'
        ordering = ['-created_at']

    def __str__(self):
        return f"Upload {self.task_id} - {self.status}"

    @property
    def progress_percentage(self):
        if self.total_files == 0:
            return 0
        return int((self.processed_files / self.total_files) * 100)
