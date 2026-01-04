from django import forms
from apps.core.models import Dataset
from django.core.validators import FileExtensionValidator
from django.forms import ModelChoiceField


class DataUploadForm(forms.Form):
    """Form for uploading gesture training data (ZIP with images)."""
    data_file = forms.FileField(
        help_text="Upload a ZIP file containing ONLY folders of images (e.g. 'like/', 'stop/').",
        validators=[FileExtensionValidator(['zip'])]
    )
    dataset_version = forms.CharField(
        max_length=50,
        required=True,
        label='Dataset Version',
        help_text='Enter a version identifier for this dataset (e.g., v1.0, 2024-01-15)'
    )


class TrainingConfigForm(forms.Form):
    """Form for configuring a new training run."""
    dataset_version = forms.ModelChoiceField(
        queryset=Dataset.objects.all(),
        label='Dataset Version',
        help_text='Select which dataset version to use for training',
        empty_label="Choose a dataset version"
    )
    description = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        required=False,
        label='Description',
        help_text='Describe what makes this training run different'
    )
    epochs = forms.IntegerField(
        initial=30,
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
        min_value=0.00001,
        max_value=1.0,
        label='Learning Rate',
    )


class ModelDeploymentForm(forms.Form):
    """Form for activating a model version."""
    confirm = forms.BooleanField(
        required=True,
        label='I confirm this activation',
    )
    notes = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 2}),
        required=False,
        label='Activation Notes'
    )

