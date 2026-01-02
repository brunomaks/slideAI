from django import forms
from apps.core.models import ModelVersion, TrainingRun
from django.core.validators import FileExtensionValidator


class DataUploadForm(forms.Form):
    """Form for uploading gesture training data (ZIP with images)."""
    data_file = forms.FileField(
        help_text="Upload a ZIP file containing ONLY folders of images (e.g. 'like/', 'stop/').",
        validators=[FileExtensionValidator(['zip'])]
    )


class TrainingConfigForm(forms.Form):
    """Form for configuring a new training run."""
    version_name = forms.CharField(
        max_length=100,
        required=False,
        label='Version Name (optional)',
        help_text='Leave empty to auto-generate timestamp-based name'
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        required=False,
        label='Description',
        help_text='Describe what makes this training run different'
    )
    
    epochs = forms.IntegerField(
        initial=15,
        min_value=1,
        max_value=100,
        label='Epochs'
    )
    batch_size = forms.IntegerField(
        initial=32,
        min_value=8,
        max_value=128,
        label='Batch Size'
    )


class ModelDeploymentForm(forms.Form):
    """Form for activating a model version."""
    confirm = forms.BooleanField(
        required=True,
        label='I confirm this activation',
        help_text='This will replace the currently active model'
    )
    notes = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2}),
        required=False,
        label='Activation Notes'
    )

