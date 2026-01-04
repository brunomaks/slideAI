"""Training service for managing model training runs.

This implementation calls the ML Training API service via HTTP.
The ml-training container runs persistently and accepts training requests.
"""
import os
import uuid
import requests
from datetime import datetime
from django.utils import timezone
from django.conf import settings
from pathlib import Path

from apps.core.models import TrainingRun, ModelVersion


# URL for the ML Training API service
ML_TRAINING_API_URL = os.getenv('ML_TRAINING_API_URL', 'http://ml-training:8003')



class TrainingService:
    """Service for initiating and managing model training via HTTP API."""
    def __init__(self):
        self.api_url = ML_TRAINING_API_URL

    def start_training(self, config: dict, user):
        """
        Start a new training run by calling the ML Training API.

        Args:
            config: Dictionary with training configuration
            user: User who initiated the training

        Returns:
            TrainingRun instance
        """
        # Create version name if not provided
        dataset = config['dataset_version']
        
        dataset_version = dataset.version

        version_name = str(dataset_version) + str(timezone.now().strftime("_%Y%m%d_%H%M%S"))

        sample_count = dataset.validated_preprocessed_samples
        if sample_count == 0:
            raise ValueError("There are 0 samples in the dataset.")
        
        # Prepare API request payload
        api_payload = {
            'epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'dataset_version': dataset_version,
            'version_name': version_name
        }

        # Call the ML Training API
        try:
            response = requests.post(
                f"{self.api_url}/train",
                json=api_payload,
                timeout=30
            )
            response.raise_for_status()
            api_result = response.json()
            job_id = api_result.get('job_id')
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to ML Training API. "
                "Make sure ml-training service is running: docker compose up -d"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to start training: {e}")

        # Create training run record
        training_run = TrainingRun.objects.create(
            run_id=job_id,
            status='running',
            started_by=user,
            started_at=timezone.now(),
            config={
                'epochs': config['epochs'],
                'batch_size': config['batch_size'],
                'learning_rate': config['learning_rate'],
                'validation_split': config.get('validation_split', 0.2),
                'version_name': version_name,
                'description': config.get('description', ''),
                'api_job_id': job_id,
            }
        )

        return training_run

    def cancel_training(self, training_run: TrainingRun):
        """
        Cancel a running training job.

        Args:
            training_run: TrainingRun instance to cancel
        """
        if training_run.status not in ['pending', 'running']:
            raise ValueError(f"Cannot cancel training in status: {training_run.status}")

        # Note: The current API doesn't support cancellation, so we just mark it locally
        training_run.status = 'cancelled'
        training_run.completed_at = timezone.now()
        training_run.save()

    def check_training_status(self, training_run: TrainingRun):
        """
        Check the status of a training run by querying the ML Training API.

        Args:
            training_run: TrainingRun instance

        Returns:
            Updated status
        """
        if training_run.status not in ['running', 'pending']:
            return training_run.status

        job_id = training_run.run_id

        try:
            response = requests.get(
                f"{self.api_url}/train/{job_id}",
                timeout=10
            )

            if response.status_code == 404:
                training_run.status = 'failed'
                training_run.completed_at = timezone.now()
                training_run.error_message = 'Training job missing from ML API.'
                training_run.save()
                return training_run.status

            response.raise_for_status()
            job_data = response.json()

            api_status = job_data.get('status', 'pending')

            if api_status == 'completed':
                training_run.status = 'completed'
                training_run.completed_at = timezone.now()
                training_run.logs = job_data.get('stdout', '')
                metrics = job_data.get('metrics')
                if metrics:
                    training_run.final_metrics = metrics
                try:
                    self._link_model_version(training_run, metrics)
                except Exception as e:
                    training_run.error_message = f"Training completed but model linking failed: {e}"
                training_run.save()

            elif api_status == 'failed':
                training_run.status = 'failed'
                training_run.completed_at = timezone.now()
                training_run.error_message = job_data.get('error', 'Unknown error')
                training_run.logs = job_data.get('stderr', '')
                training_run.save()
            elif api_status == 'running':
                training_run.status = 'running'
                if not training_run.started_at:
                    training_run.started_at = timezone.now()
                # Try to fetch current logs
                try:
                    logs_response = requests.get(
                        f"{self.api_url}/train/{job_id}/logs",
                        timeout=10
                    )
                    if logs_response.status_code == 200:
                        logs_data = logs_response.json()
                        training_run.logs = logs_data.get('stdout', '') + logs_data.get('stderr', '')
                except requests.exceptions.RequestException:
                    pass
                training_run.save()

        except requests.exceptions.RequestException:
            # API unavailable, keep current status
            pass

        return training_run.status

    def _link_model_version(self, training_run: TrainingRun, metrics=None):
        """
        Link training run to the created model version.

        Args:
            training_run: TrainingRun instance
        """
        version_name = training_run.config.get('version_name')

        if not version_name:
            return

        # Expected artifact file name produced by trainer
        # train.py saves to: gesture_model_{version}.keras
        model_filename = f"gesture_model_{version_name}.keras"
        version_id = Path(model_filename).stem

        # Try to find an existing ModelVersion first
        model_version = ModelVersion.objects.filter(version_id=version_id).first()

        if not model_version:
            # Determine if this model was set active by trainer (active_model.json)
            model_path = Path(settings.MODEL_PATH)
            active_file = model_path / 'active_model.json'
            is_active = False
            if active_file.exists():
                try:
                    import json
                    with open(active_file, 'r') as f:
                        active_data = json.load(f)
                    is_active = (active_data.get('model_file') == model_filename)
                except Exception:
                    is_active = False

            if is_active:
                ModelVersion.objects.filter(is_active=True).update(is_active=False)

            model_version = ModelVersion.objects.create(
                version_id=version_id,
                model_file=model_filename,
                framework='tensorflow',
                training_date=timezone.now(),
                # Dataset info
                train_dataset_size=(metrics or {}).get('dataset', {}).get('train_count') or 0,
                test_dataset_size=(metrics or {}).get('dataset', {}).get('test_count') or 0,
                validation_dataset_size=(metrics or {}).get('dataset', {}).get('validation_count') or 0,
                # Hyperparameters
                # Create a new ModelVersion entry from training contexs
                learning_rate=training_run.config["learning_rate"],
                epochs=(metrics or {}).get('config', {}).get('epochs') or training_run.config.get('epochs') or 0,
                batch_size=(metrics or {}).get('config', {}).get('batch_size') or training_run.config.get('batch_size') or 0,
                # Metrics - convert from 0-1 scale to percentage (0-100)
                train_accuracy=((metrics or {}).get('train', {}).get('accuracy') or 0) * 100,
                validation_accuracy=((metrics or {}).get('validation', {}).get('accuracy') or 0) * 100,
                test_accuracy=((metrics or {}).get('test', {}).get('accuracy') or 0) * 100,
                train_loss=(metrics or {}).get('train', {}).get('loss'),
                validation_loss=(metrics or {}).get('validation', {}).get('loss'),
                test_loss=(metrics or {}).get('test', {}).get('loss'),
                test_precision=((metrics or {}).get('test', {}).get('precision') or 0) * 100,
                test_recall=((metrics or {}).get('test', {}).get('recall') or 0) * 100,
                test_f1_score=((metrics or {}).get('test', {}).get('f1_score') or 0) * 100,
                confusion_matrix=(metrics or {}).get('test', {}).get('confusion_matrix'),
                is_active=is_active,
                description=training_run.config.get('description', ''),
            )

        training_run.model_version = model_version
        if not training_run.final_metrics:
            training_run.final_metrics = {
                'test_accuracy': model_version.test_accuracy,
                'validation_accuracy': model_version.validation_accuracy,
                'train_accuracy': model_version.train_accuracy,
            }
