# Contributors:
# - Mahmoud

"""Model management service for deploying, rolling back, and comparing models."""
from django.utils import timezone
from django.conf import settings
from pathlib import Path
import os
import json
import sqlite3
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
                "Cannot deploy model: This model was trained with 0 samples. "
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
        
        # Update the active_model.json file (read by inference service)
        model_path = Path(settings.MODEL_PATH)
        active_model_file = model_path / 'active_model.json'
        active_data = {
            "model_file": model.model_file,
            "class_names": ModelManager._get_class_names_from_model(model)
        }
        with open(active_model_file, 'w') as f:
            json.dump(active_data, f, indent=2)
        
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

    @staticmethod
    def delete_model(model: ModelVersion):
        """
        Delete a model version and its associated file.
        
        Args:
            model: ModelVersion instance to delete
            
        Raises:
            ValueError: If trying to delete the active model
        """
        if model.is_active:
            raise ValueError("Cannot delete the active model. Please deploy another model first.")
            
        # Delete file from filesystem
        if model.model_file:
            model_path = Path(settings.MODEL_PATH) / model.model_file
            try:
                if model_path.exists():
                    os.remove(model_path)
            except Exception as e:
                # Log error but proceed with DB deletion
                print(f"Error deleting model file {model_path}: {e}")
                
        # Delete from database
        model.delete()

    @staticmethod
    def _get_class_names_from_model(model: ModelVersion):
        """
        Get the class names for a model by reading from landmarks database.
        
        The actual class names are determined at training time from the database.
        We read them from the gestures_processed table.
        """
        try:
            db_path = settings.DATABASES['landmarks']['NAME']
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT gesture FROM gestures_processed ORDER BY gesture")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            # Fallback: return empty list (inference will fail but won't crash deploy)
            return []
