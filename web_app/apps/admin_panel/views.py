from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count, Avg, Q
from datetime import datetime, timedelta
import os
import zipfile
import shutil
from pathlib import Path

from apps.core.models import ModelVersion, Prediction, TrainingRun, ImageMetadata
from .services.model_manager import ModelManager


def is_staff_or_superuser(user):
    """Check if user is staff or superuser."""
    return user.is_staff or user.is_superuser


@login_required
@user_passes_test(is_staff_or_superuser)
def dashboard(request):
    """Admin dashboard with overview of system status."""
    # Get active model
    active_model = ModelVersion.objects.filter(is_active=True).first()
    
    # Get recent predictions
    recent_predictions = Prediction.objects.select_related('model_version').order_by('-created_at')[:10]
    
    # Get prediction stats (last 24 hours)
    yesterday = timezone.now() - timedelta(hours=24)
    predictions_24h = Prediction.objects.filter(created_at__gte=yesterday).count()
    avg_confidence_24h = Prediction.objects.filter(
        created_at__gte=yesterday
    ).aggregate(avg_conf=Avg('confidence'))['avg_conf'] or 0
    
    # Get training runs
    active_training = TrainingRun.objects.filter(status='running').first()
    recent_trainings = TrainingRun.objects.order_by('-created_at')[:5]
    
    # Get model count
    total_models = ModelVersion.objects.count()
    
    # Get dataset stats
    train_images = ImageMetadata.objects.filter(dataset_type='train').count()
    test_images = ImageMetadata.objects.filter(dataset_type='test').count()
    
    # Prediction distribution by class (last 24h)
    class_distribution = Prediction.objects.filter(
        created_at__gte=yesterday
    ).values('predicted_class').annotate(count=Count('id')).order_by('-count')
    
    # Calculate percentages
    class_dist_with_percentage = []
    for item in class_distribution:
        percentage = (item['count'] * 100 / predictions_24h) if predictions_24h > 0 else 0
        class_dist_with_percentage.append({
            'predicted_class': item['predicted_class'],
            'count': item['count'],
            'percentage': round(percentage, 1)
        })
    
    context = {
        'active_model': active_model,
        'recent_predictions': recent_predictions,
        'predictions_24h': predictions_24h,
        'avg_confidence_24h': avg_confidence_24h,
        'active_training': active_training,
        'recent_trainings': recent_trainings,
        'total_models': total_models,
        'train_images': train_images,
        'test_images': test_images,
        'class_distribution': class_dist_with_percentage,
    }
    
    return render(request, 'admin_panel/dashboard.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def models_list(request):
    """List all model versions."""
    models = ModelVersion.objects.all()
    
    context = {
        'models': models,
    }
    
    return render(request, 'admin_panel/models_list.html', context)

