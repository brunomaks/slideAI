from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'admin_panel'

urlpatterns = [
    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='admin_panel/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/admin/login/'), name='logout'),

    # Dashboard
    path('', views.dashboard, name='dashboard'),

    # Model Management (minimal)
    path('models/', views.models_list, name='models_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/<int:model_id>/deploy/', views.deploy_model, name='deploy_model'),
    # Data Upload
    path('data/upload/', views.upload_data, name='upload_data'),
    path('data/images/', views.view_images, name='view_images'),
]
