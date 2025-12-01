from django.urls import path
from .views import flip_view


urlpatterns = [
path('', flip_view, name='flip'),
]
