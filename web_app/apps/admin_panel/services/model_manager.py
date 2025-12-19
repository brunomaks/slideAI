"""Model management service for deploying, rolling back, and comparing models."""
from django.utils import timezone
from django.conf import settings
from pathlib import Path
from apps.core.models import ModelVersion


class ModelManager:
    """Service for managing model versions and deployment."""
    
    @staticmethod
    def deploy_model(model: ModelVersion, user, notes: str = ''):
        """
        Deploy a model version as the active model.
        
        Args:
            model: ModelVersion instance to deploy
            user: User performing the deployment
            notes: Optional deployment notes
        """
        # Validate that the model was trained with actual data
        if model.train_dataset_size is None or model.train_dataset_size == 0:
            raise ValueError(
                "Cannot deploy model: This model was trained with 0 training images. "
                "Models must be trained with data before deployment."
            )
        
        # Deactivate all other models
        ModelVersion.objects.filter(is_active=True).update(is_active=False)
        
        # Activate this model
        model.is_active = True
        model.deployment_date = timezone.now()
        model.deployed_by = user
        if notes:
            model.notes = f"{model.notes}\n\n[{timezone.now()}] Deployment: {notes}" if model.notes else notes
        model.save()
        
        # Update the active_model.txt file
        model_path = Path(settings.MODEL_PATH)
        active_model_file = model_path / 'active_model.txt'
        active_model_file.write_text(model.model_file)
        
        return model
    
    @staticmethod
    def rollback_to_model(model: ModelVersion, user):
        """
        Rollback to a previous model version.
        
        Args:
            model: ModelVersion instance to rollback to
            user: User performing the rollback
        """
        return ModelManager.deploy_model(
            model, 
            user, 
            notes=f"Rollback initiated by {user.username}"
        )
    
    @staticmethod
    def get_active_model():
        """Get the currently active model."""
        return ModelVersion.objects.filter(is_active=True).first()

