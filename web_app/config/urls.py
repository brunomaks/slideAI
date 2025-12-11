from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('offer/', include('apps.main.urls')),
    path('resize/', include('apps.resize.urls')),
]
