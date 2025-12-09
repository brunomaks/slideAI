from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('offer/', include('apps.main.urls')),
    path('grayscale/', include('apps.grayscale.urls')),
    path('flip/', include('apps.flip.urls')),
]
