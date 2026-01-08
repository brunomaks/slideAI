# Contributors:
# - Mahmoud

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
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

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

from pathlib import Path

from apps.core.models import ModelVersion, Prediction, TrainingRun, Dataset
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
    stats = Dataset.get_latest_statistics()
    total_samples = stats['total_samples']
    
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
        'total_samples': total_samples,
        'class_distribution': class_dist_with_percentage,
    }

    return render(request, 'admin_panel/dashboard.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def models_list(request):
    """List all model versions."""
    # Sync any completed training runs that might not have models linked yet
    # This handles the case where user navigates directly to models without visiting training status
    service = TrainingService()
    pending_sync_runs = TrainingRun.objects.filter(
        status__in=['running', 'pending'],
    ).select_related('model_version')

    for run in pending_sync_runs:
        try:
            service.check_training_status(run)
        except Exception:
            pass  # Don't block page load for API errors

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

    # Generate confusion matrix image if available
    cm = model.confusion_matrix
    labels = model.class_labels
    if model.confusion_matrix and model.class_labels:
        cm = np.array(model.confusion_matrix)
        cm_image = plot_confusion_matrix(cm, labels)
    else:
        cm_image = None

    context = {
        'model': model,
        'predictions': predictions,
        'cm_image': cm_image,
        'prediction_count': prediction_count,
        'avg_confidence': avg_confidence,
        'class_dist': class_dist_with_percentage,
    }

    return render(request, 'admin_panel/model_detail.html', context)


def plot_confusion_matrix(cm, labels, fig_bg_color='#262c49', ax_bg_color='#2c3355', text_color='white'):
    # Create custom colormap
    custom_cmap = LinearSegmentedColormap.from_list('dark_theme', ['#2c3355', '#6f5bdc'])
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(fig_bg_color)
    ax.set_facecolor(ax_bg_color)

    # Create heatmap
    heatmap = sns.heatmap(
        cm, annot=True, fmt='d', cmap=custom_cmap,
        xticklabels=labels, yticklabels=labels,
        annot_kws={"color": text_color, "weight": "bold"},
        ax=ax, cbar=True, linewidths=0.5, linecolor='#1f2233'
    )
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(cbar.ax.get_yticklabels(), color=text_color)
    if cbar.ax.get_ylabel():
        cbar.ax.yaxis.label.set_color(text_color)

    # Labels and title
    ax.set_xlabel('Predicted', color=text_color)
    ax.set_ylabel('Actual', color=text_color)
    ax.set_title('Confusion Matrix', color=text_color)
    ax.tick_params(colors=text_color)

    # Save to buffer
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor=fig_bg_color)
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@login_required
@user_passes_test(is_staff_or_superuser)
def deploy_model(request, model_id):
    """Activate a model version as the active model."""
    model = get_object_or_404(ModelVersion, id=model_id)

    if request.method == 'POST':
        form = ModelDeploymentForm(request.POST)
        if form.is_valid():
            try:
                ModelManager.deploy_model(model, request.user, form.cleaned_data.get('notes', ''))
                messages.success(request, f'Model {model.version_id} activated successfully!')
                return redirect('admin_panel:models_list')
            except Exception as e:
                messages.error(request, f'Activation failed: {str(e)}')
    else:
        form = ModelDeploymentForm()

    context = {
        'model': model,
        'form': form,
    }

    return render(request, 'admin_panel/deploy_model.html', context)


@login_required
@user_passes_test(is_staff_or_superuser)
def upload_data(request):
    """Upload new labeled training data."""
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploader = DataUploader()
                data_file = request.FILES['data_file']
                dataset_version = form.cleaned_data['dataset_version']

                counts = uploader.handle_upload(data_file, dataset_version=dataset_version, user=request.user)

                messages.success(
                    request, 
                    f"Successfully processed {counts['total']} raw samples."
                )
                return redirect('admin_panel:view_dataset')
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
def view_dataset(request):
    """View training dataset statistics (Landmarks)."""

    dataset_versions = Dataset.objects.all()

    selected_version = request.GET.get('version')
    if selected_version:
        current_dataset = dataset_versions.filter(version=selected_version).first()
    else:
        current_dataset = dataset_versions.first()
    if not current_dataset:
        return render(request, 'admin_panel/view_dataset.html', {
            'dataset_versions': [],
        })
    
    stats = Dataset.get_statistics_for_version(current_dataset.version)
    # {'label_stats': {'like': 1464, 'stop': 1599, 'two_up_inverted': 1525}, 'total_samples': 4588}
    
    total_samples = stats['total_samples']
    label_counts = stats['label_stats']

    label_stats = [
        {'label': label, 'count': count}
        for label, count in label_counts.items()
    ]

    labels = sorted([item for item in label_counts])

    context = {
        'dataset_versions': dataset_versions,
        'current_dataset': current_dataset,
        'total_samples': total_samples,
        'labels': labels,
        'label_stats': label_stats
    }

    return render(request, 'admin_panel/view_dataset.html', context)


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
def performance_overview(request):
    """Assess current model performance."""
    days = int(request.GET.get('days', 1))
    days = max(1, min(days, 30))
    since = timezone.now() - timedelta(days=days)

    active_model = ModelVersion.objects.filter(is_active=True).first()

    preds_qs = Prediction.objects.filter(created_at__gte=since)
    total_preds = preds_qs.count()
    avg_conf = preds_qs.aggregate(avg=Avg('confidence'))['avg'] or 0

    class_dist = preds_qs.values('predicted_class').annotate(count=Count('id')).order_by('-count')
    dist = []
    for item in class_dist:
        pct = (item['count'] * 100 / total_preds) if total_preds else 0
        dist.append({
            'predicted_class': item['predicted_class'],
            'count': item['count'],
            'percentage': round(pct, 1)
        })

    context = {
        'active_model': active_model,
        'days': days,
        'total_preds': total_preds,
        'avg_confidence': avg_conf,
        'class_distribution': dist,
    }
    return render(request, 'admin_panel/performance_overview.html', context)


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
                messages.success(
                    request,
                    f"Training run {training_run.run_id} created! "
                    f"View the Training Status page for the command to run."
                )
                return redirect('admin_panel:training_status')
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

@login_required
@user_passes_test(is_staff_or_superuser)
@require_POST
def delete_model(request, model_id):
    """Delete a model version."""
    model = get_object_or_404(ModelVersion, id=model_id)
    version_id = model.version_id

    try:
        ModelManager.delete_model(model)
        messages.success(request, f"Model {version_id} deleted successfully.")
    except Exception as e:
        messages.error(request, f"Failed to delete model {version_id}: {str(e)}")

    return redirect('admin_panel:models_list')


@login_required
@user_passes_test(is_staff_or_superuser)
@require_POST
def delete_training_run(request, run_id):
    """Delete a training run record."""
    run = get_object_or_404(TrainingRun, id=run_id)
    run_id_str = run.run_id

    # Only allow deleting completed, failed, or cancelled runs
    if run.status in ['pending', 'running']:
        messages.error(request, f"Cannot delete a {run.status} training run. Cancel it first.")
        return redirect('admin_panel:training_status')

    try:
        run.delete()
        messages.success(request, f"Training run {run_id_str} deleted successfully.")
    except Exception as e:
        messages.error(request, f"Failed to delete training run: {str(e)}")

    return redirect('admin_panel:training_status')
