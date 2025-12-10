from django.urls import path
from .views import resize_view


urlpatterns = [
path('', resize_view, name='resize'),
]
