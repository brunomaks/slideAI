from django import forms
from apps.core.models import ModelVersion, TrainingRun


class DataUploadForm(forms.Form):
    """Form for uploading labeled training data."""
    data_file = forms.FileField(
        label='Upload Data File',
        help_text='Accepted formats: ZIP containing images in labeled folders, or CSV with metadata',
        widget=forms.FileInput(attrs={'accept': '.zip,.csv'})
    )
    dataset_type = forms.ChoiceField(
        choices=[('train', 'Training'), ('test', 'Test'), ('validation', 'Validation')],
        initial='train',
        label='Dataset Type'
    )
    label = forms.CharField(
        max_length=50,
        required=False,
        label='Gesture Label (optional)',
        help_text='Leave empty for multi-class uploads (ZIP with folders)'
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
    image_size = forms.IntegerField(
        initial=128,
        min_value=64,
        max_value=512,
        label='Image Size (pixels)'
    )
    
    validation_split = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.5,
        label='Validation Split Ratio'
    )

