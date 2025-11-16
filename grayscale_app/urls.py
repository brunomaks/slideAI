from django.urls import path
from .views import grayscale_view

urlpatterns = [
    path('', grayscale_view),
]
