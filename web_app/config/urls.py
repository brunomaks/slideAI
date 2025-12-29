from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('django-admin/', admin.site.urls),  # Django built-in admin (development only)
    path('admin/', include('apps.admin_panel.urls')),
    path('api/', include('apps.main.urls')),
    path('resize/', include('apps.resize.urls')),
]
