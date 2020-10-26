from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="AI-home"),
    path('about/', views.about, name="AI-about"),
    path('test/', views.test, name="test"),
]