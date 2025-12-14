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
from .forms import DataUploadForm, TrainingConfigForm, ModelDeploymentForm
from .services.model_manager import ModelManager
from .services.training_service import TrainingService
from .services.data_uploader import DataUploader


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


@login_required
@user_passes_test(is_staff_or_superuser)
def model_detail(request, model_id):
    """View details of a specific model."""
    model = get_object_or_404(ModelVersion, id=model_id)
    
    # Get predictions made by this model
    predictions = model.predictions.order_by('-created_at')[:100]
    prediction_count = model.predictions.count()
    avg_confidence = model.predictions.aggregate(avg=Avg('confidence'))['avg'] or 0
    
    # Get class distribution
    class_dist = model.predictions.values('predicted_class').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Calculate percentages
    class_dist_with_percentage = []
    for item in class_dist:
        percentage = (item['count'] * 100 / prediction_count) if prediction_count > 0 else 0
        class_dist_with_percentage.append({
            'predicted_class': item['predicted_class'],
            'count': item['count'],
            'percentage': round(percentage, 1)
        })
    
    context = {
        'model': model,
        'predictions': predictions,
        'prediction_count': prediction_count,
        'avg_confidence': avg_confidence,
        'class_dist': class_dist_with_percentage,
    }
    
    return render(request, 'admin_panel/model_detail.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def deploy_model(request, model_id):
    """Deploy a model version as the active model."""
    model = get_object_or_404(ModelVersion, id=model_id)
    
    if request.method == 'POST':
        form = ModelDeploymentForm(request.POST)
        if form.is_valid():
            try:
                ModelManager.deploy_model(model, request.user, form.cleaned_data.get('notes', ''))
                messages.success(request, f'Model {model.version_id} deployed successfully!')
                return redirect('admin_panel:models_list')
            except Exception as e:
                messages.error(request, f'Deployment failed: {str(e)}')
    else:
        form = ModelDeploymentForm()
    
    context = {
        'model': model,
        'form': form,
    }
    
    return render(request, 'admin_panel/deploy_model.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def rollback_model(request, model_id):
    """Rollback to a previous model version."""
    model = get_object_or_404(ModelVersion, id=model_id)
    
    if request.method == 'POST':
        try:
            ModelManager.rollback_to_model(model, request.user)
            messages.success(request, f'Rolled back to model {model.version_id}!')
            return redirect('admin_panel:models_list')
        except Exception as e:
            messages.error(request, f'Rollback failed: {str(e)}')
            return redirect('admin_panel:models_list')
    
    context = {
        'model': model,
    }
    
    return render(request, 'admin_panel/rollback_model.html', context)



@login_required
@user_passes_test(is_staff_or_superuser)
def upload_data(request):
    """Upload new labeled training data."""
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploader = DataUploader()
                result = uploader.handle_upload(
                    request.FILES['data_file'],
                    form.cleaned_data['dataset_type'],
                    form.cleaned_data.get('label')
                )
                messages.success(
                    request, 
                    f"Successfully uploaded {result['count']} images for {result['dataset_type']} dataset."
                )
                return redirect('admin_panel:view_images')
            except Exception as e:
                messages.error(request, f'Upload failed: {str(e)}')
    else:
        form = DataUploadForm()
    
    context = {
        'form': form,
    }
    
    return render(request, 'admin_panel/upload_data.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def view_images(request):
    """View uploaded images by label."""
    label_filter = request.GET.get('label', '')
    dataset_type = request.GET.get('dataset_type', 'train')
    
    images_query = ImageMetadata.objects.filter(dataset_type=dataset_type)
    
    if label_filter:
        images_query = images_query.filter(label=label_filter)
    
    images = images_query.order_by('-created_at')[:100]
    
    # Get available labels
    labels = ImageMetadata.objects.values_list('label', flat=True).distinct().order_by('label')
    
    # Get stats by label
    label_stats = ImageMetadata.objects.filter(dataset_type=dataset_type).values('label').annotate(
        count=Count('id')
    ).order_by('label')
    
    context = {
        'images': images,
        'labels': labels,
        'label_stats': label_stats,
        'selected_label': label_filter,
        'dataset_type': dataset_type,
    }
    
    return render(request, 'admin_panel/view_images.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def compare_models(request):
    """Compare active model with a selected candidate model."""
    active_model = ModelVersion.objects.filter(is_active=True).first()
    candidate_id = request.GET.get('candidate')
    candidate = None
    models = ModelVersion.objects.all().order_by('-created_at')

    if candidate_id:
        candidate = get_object_or_404(ModelVersion, id=candidate_id)

    context = {
        'active_model': active_model,
        'candidate': candidate,
        'models': models,
    }
    return render(request, 'admin_panel/compare_models.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def start_training(request):
    """Start a new training run."""
    if request.method == 'POST':
        form = TrainingConfigForm(request.POST)
        if form.is_valid():
            try:
                service = TrainingService()
                training_run = service.start_training(form.cleaned_data, request.user)
                messages.success(request, f"Training started: {training_run.run_id}")
                return redirect('admin_panel:dashboard')
            except Exception as e:
                messages.error(request, f"Failed to start training: {e}")
    else:
        form = TrainingConfigForm()

    context = {
        'form': form,
    }
    return render(request, 'admin_panel/start_training.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def training_status(request):
    """List recent training runs with status and allow refresh."""
    service = TrainingService()
    runs = TrainingRun.objects.order_by('-created_at')[:20]

    # Optionally refresh statuses for running/pending runs
    for run in runs:
        if run.status in ['running', 'pending']:
            try:
                service.check_training_status(run)
            except Exception:
                pass

    context = {
        'runs': runs,
    }
    return render(request, 'admin_panel/training_status.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def cancel_training(request, run_id):
    """Cancel a specific training run."""
    run = get_object_or_404(TrainingRun, id=run_id)
    try:
        service = TrainingService()
        service.cancel_training(run)
        messages.success(request, f"Cancelled training: {run.run_id}")
    except Exception as e:
        messages.error(request, f"Failed to cancel training: {e}")
    return redirect('admin_panel:training_status')

