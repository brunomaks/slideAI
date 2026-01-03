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
    path('models/<int:model_id>/delete/', views.delete_model, name='delete_model'),
    path('models/compare/', views.compare_models, name='compare_models'),
    path('models/performance/', views.performance_overview, name='performance_overview'),

    # Data Upload
    path('data/upload/', views.upload_data, name='upload_data'),
    path('data/dataset/', views.view_dataset, name='view_dataset'),

    # Training
    path('training/start/', views.start_training, name='start_training'),
    path('training/status/', views.training_status, name='training_status'),
    path('training/cancel/<int:run_id>/', views.cancel_training, name='cancel_training'),
    path('training/delete/<int:run_id>/', views.delete_training_run, name='delete_training_run'),
]
