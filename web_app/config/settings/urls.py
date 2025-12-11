from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
path('admin/', admin.site.urls),
path('offer/', include('main.urls')),
path('grayscale/', include('grayscale.urls')),
path('flip/', include('flip.urls')),
]

# serve media files in development
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
