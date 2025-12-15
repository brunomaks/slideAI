#!/usr/bin/env python
import os
import django
from pathlib import Path
from datetime import datetime

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
            training_date = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
        except:
            training_date = datetime.now()
        
        ModelVersion.objects.create(
            version_id=version_id,
            model_file=model_file.name,
            framework='tensorflow',
            training_date=training_date,
            test_accuracy=90.0,
            is_active=False
        )
        print(f"Registered: {version_id}")
    
    active_model_file = model_path / 'active_model.txt'
    if active_model_file.exists():
        version_id = Path(active_model_file.read_text().strip()).stem
        model = ModelVersion.objects.filter(version_id=version_id).first()
        if model:
            ModelVersion.objects.filter(is_active=True).update(is_active=False)
            model.is_active = True
            model.save()


if __name__ == '__main__':
    create_admin_user()
    register_existing_models()
