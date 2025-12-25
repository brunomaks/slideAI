from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'change-me-in-production')

FORCE_SCRIPT_NAME = os.environ.get('SCRIPT_NAME', '/api')

# Static files for production
STATIC_ROOT = BASE_DIR / 'staticfiles'

# CORS settings for production
CORS_ALLOWED_ORIGINS = [
    origin.strip() 
    for origin in os.environ.get('CORS_ALLOWED_ORIGINS', '').split(',') 
    if origin.strip()
]

# CSRF settings for production
CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get('CSRF_TRUSTED_ORIGINS', '').split(',')
    if origin.strip()
]

