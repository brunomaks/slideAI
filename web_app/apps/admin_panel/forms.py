from django import forms
from apps.core.models import ModelVersion, TrainingRun


class DataUploadForm(forms.Form):
    """Form for uploading labeled training data (landmarks JSON)."""
    data_file = forms.FileField(
        label='Upload Landmarks JSON',
        help_text='JSON file containing labeled landmarks data',
        widget=forms.FileInput(attrs={'accept': '.json'})
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
    learning_rate = forms.FloatField(
        initial=0.001,
        min_value=0.0001,
        max_value=0.1,
        label='Learning Rate'
    )
    
    validation_split = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.5,
        label='Validation Split Ratio'
    )


class ModelDeploymentForm(forms.Form):
    """Form for deploying a model version."""
    confirm = forms.BooleanField(
        required=True,
        label='I confirm this deployment',
        help_text='This will replace the currently active model'
    )
    notes = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2}),
        required=False,
        label='Deployment Notes'
    )

