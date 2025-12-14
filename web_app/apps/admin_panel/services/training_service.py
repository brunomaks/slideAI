"""Training service for managing model training runs."""
import subprocess
import uuid
from datetime import datetime
from django.utils import timezone
from django.conf import settings
from pathlib import Path

from apps.core.models import TrainingRun, ModelVersion, ImageMetadata


class TrainingService:
    """Service for initiating and managing model training."""
    
    def start_training(self, config: dict, user):
        """
        Start a new training run.
        
        Args:
            config: Dictionary with training configuration
            user: User who initiated the training
            
        Returns:
            TrainingRun instance
        """
        # Generate run ID
        run_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Create version name if not provided
        version_name = config.get('version_name') or datetime.now().strftime("model_%Y%m%d_%H%M%S")
        
        # Get dataset sizes
        train_size = ImageMetadata.objects.filter(dataset_type='train').count()
        test_size = ImageMetadata.objects.filter(dataset_type='test').count()
        val_size = ImageMetadata.objects.filter(dataset_type='validation').count()
        
        # Create training run record
        training_run = TrainingRun.objects.create(
            run_id=run_id,
            status='pending',
            started_by=user,
            config={
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'image_size': config['image_size'],
                'validation_split': config.get('validation_split', 0.2),

                'version_name': version_name,
                'description': config.get('description', ''),
                'train_dataset_size': train_size,
                'test_dataset_size': test_size,
                'validation_dataset_size': val_size,
            }
        )
        
        # Start training in background (Docker container)
        try:
            self._trigger_docker_training(training_run)
            training_run.status = 'running'
            training_run.started_at = timezone.now()
            training_run.save()
        except Exception as e:
            training_run.status = 'failed'
            training_run.error_message = str(e)
            training_run.save()
            raise
        
        return training_run
    
    def _trigger_docker_training(self, training_run: TrainingRun):
        """
        Trigger training via Docker Compose.
        
        Args:
            training_run: TrainingRun instance
        """
        config = training_run.config
        
        # Build docker-compose command
        command = [
            'docker-compose',
            'run',
            '--rm',
            '-d',  # Run in detached mode
            '--name', f"training_{training_run.run_id}",
            'ml-training',
            'python', 'src/train.py',
            '--epochs', str(config['epochs']),
            '--batch-size', str(config['batch_size']),
            '--img-size', str(config['image_size']),
            '--version', config['version_name'],
        ]
        
        # Execute command
        # Note: This requires docker-compose to be accessible from Django container
        # In production, use Kubernetes Job or Celery task
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        training_run.logs = result.stdout
        training_run.save()
    
    def cancel_training(self, training_run: TrainingRun):
        """
        Cancel a running training job.
        
        Args:
            training_run: TrainingRun instance to cancel
        """
        if training_run.status != 'running':
            raise ValueError(f"Cannot cancel training in status: {training_run.status}")
        
        # Stop Docker container
        try:
            subprocess.run(
                ['docker', 'stop', f"training_{training_run.run_id}"],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError:
            pass  # Container might already be stopped
        
        training_run.status = 'cancelled'
        training_run.completed_at = timezone.now()
        training_run.save()
    
    def check_training_status(self, training_run: TrainingRun):
        """
        Check the status of a training run.
        
        Args:
            training_run: TrainingRun instance
            
        Returns:
            Updated status
        """
        if training_run.status not in ['running', 'pending']:
            return training_run.status
        
        # Check if Docker container is still running
        try:
            result = subprocess.run(
                ['docker', 'ps', '-q', '-f', f"name=training_{training_run.run_id}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if not result.stdout.strip():
                # Container stopped
                training_run.status = 'completed'
                training_run.completed_at = timezone.now()
                
                # Try to link to model version
                self._link_model_version(training_run)
                
                training_run.save()
        except Exception as e:
            training_run.status = 'failed'
            training_run.error_message = str(e)
            training_run.completed_at = timezone.now()
            training_run.save()
        
        return training_run.status
    
    def _link_model_version(self, training_run: TrainingRun):
        """
        Link training run to the created model version.
        
        Args:
            training_run: TrainingRun instance
        """
        version_name = training_run.config.get('version_name')
        
        if version_name:
            # Find the model version created by this training
            model_version = ModelVersion.objects.filter(
                version_id__contains=version_name
            ).order_by('-created_at').first()
            
            if model_version:
                training_run.model_version = model_version
                training_run.final_metrics = {
                    'test_accuracy': model_version.test_accuracy,
                    'validation_accuracy': model_version.validation_accuracy,
                    'train_accuracy': model_version.train_accuracy,
                }
