from django.urls import path
import django.urls
from . import views

app_name = 'blog'

urlpatterns = [
    path('', views.upload_file, name='upload_file'),
    path('post/<int:pk>/', views.post_detail, name='post_detail'),
    path('post/new/', views.post_new, name='post_new'),
    path('train/<alg>/', views.train, name='train')
    #path('post/upload/', views.upload_file, name='upload_file'),
]