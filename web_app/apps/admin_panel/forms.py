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


