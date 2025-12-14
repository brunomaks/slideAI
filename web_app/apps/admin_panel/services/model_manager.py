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
    
    @staticmethod
    def compare_models(model_1: ModelVersion, model_2: ModelVersion):
        """
        Compare two model versions.
        
        Args:
            model_1: First ModelVersion instance
            model_2: Second ModelVersion instance
            
        Returns:
            Dictionary with comparison data
        """
        comparison = {
            'model_1': {
                'version_id': model_1.version_id,
                'test_accuracy': model_1.test_accuracy,
                'train_accuracy': model_1.train_accuracy,
                'validation_accuracy': model_1.validation_accuracy,
                'epochs': model_1.epochs,
                'batch_size': model_1.batch_size,
                'learning_rate': model_1.learning_rate,
                'train_dataset_size': model_1.train_dataset_size,
                'test_dataset_size': model_1.test_dataset_size,
                'training_date': model_1.training_date,
                'is_active': model_1.is_active,
                'prediction_count': model_1.predictions.count(),
            },
            'model_2': {
                'version_id': model_2.version_id,
                'test_accuracy': model_2.test_accuracy,
                'train_accuracy': model_2.train_accuracy,
                'validation_accuracy': model_2.validation_accuracy,
                'epochs': model_2.epochs,
                'batch_size': model_2.batch_size,
                'learning_rate': model_2.learning_rate,
                'train_dataset_size': model_2.train_dataset_size,
                'test_dataset_size': model_2.test_dataset_size,
                'training_date': model_2.training_date,
                'is_active': model_2.is_active,
                'prediction_count': model_2.predictions.count(),
            },
            'differences': {
                'accuracy_diff': model_2.test_accuracy - model_1.test_accuracy,
                'epochs_diff': model_2.epochs - model_1.epochs,
                'dataset_size_diff': model_2.train_dataset_size - model_1.train_dataset_size,
            }
        }
        
        return comparison
    
    @staticmethod
    def get_model_statistics(model: ModelVersion):
        """
        Get detailed statistics for a model.
        
        Args:
            model: ModelVersion instance
            
        Returns:
            Dictionary with model statistics
        """
        predictions = model.predictions.all()
        
        return {
            'total_predictions': predictions.count(),
            'avg_confidence': predictions.aggregate(avg=models.Avg('confidence'))['avg'] or 0,
            'class_distribution': predictions.values('predicted_class').annotate(
                count=models.Count('id')
            ).order_by('-count'),
        }
