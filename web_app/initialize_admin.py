#!/usr/bin/env python
# Contributors:
# - Mahmoud

import os
import json
import django
from pathlib import Path
from datetime import datetime
from django.utils import timezone

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.development')
django.setup()

from django.contrib.auth import get_user_model
from apps.core.models import ModelVersion
from django.conf import settings

User = get_user_model()


def create_admin_user():
    if not User.objects.filter(is_superuser=True).exists():
        User.objects.create_superuser(username='admin', email='admin@slideai.local', password='admin123')
        print("Admin user created: admin / admin123")
    else:
        print("Admin user already exists")


def register_existing_models():
    model_path = Path(settings.MODEL_PATH)
    if not model_path.exists():
        return
    
    for model_file in model_path.glob("*.keras"):
        version_id = model_file.stem
        if ModelVersion.objects.filter(version_id=version_id).exists():
            continue
        
        try:
            parts = version_id.split('_')
            naive_dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
            training_date = timezone.make_aware(naive_dt, timezone.get_current_timezone())
        except:
            training_date = timezone.now()

        # Sensible defaults for required fields
        default_epochs = 10
        default_batch = 32
        train_data_size = 4588 # default number of entries for dataset featuring (like, stop and two_up_inverted)

        ModelVersion.objects.create(
            version_id=version_id,
            model_file=model_file.name,
            framework='tensorflow',
            training_date=training_date,
            # Hyperparameters (defaults; actual may differ)
            epochs=default_epochs,
            batch_size=default_batch,
            train_dataset_size=train_data_size,
            # Performance metrics
            test_accuracy=0.0,
            is_active=False
        )
        print(f"Registered: {version_id}")
    
    # Check active model from active_model.json
    active_model_file = model_path / 'active_model.json'
    if active_model_file.exists():
        try:
            with open(active_model_file, 'r') as f:
                active_data = json.load(f)
            model_filename = active_data.get('model_file', '')
            version_id = Path(model_filename).stem
            model = ModelVersion.objects.filter(version_id=version_id).first()
            if model:
                ModelVersion.objects.filter(is_active=True).update(is_active=False)
                model.is_active = True
                model.save()
        except Exception:
            pass


if __name__ == '__main__':
    create_admin_user()
    register_existing_models()
