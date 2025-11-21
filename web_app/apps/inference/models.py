from django.db import models
from apps.core.models import BaseModel

class Prediction(BaseModel):
    input_features = models.JSONField()
    prediction = models.FloatField()

    def __str__(self):
        return f"Prediction {self.id}"
