"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from djangoProject import views

urlpatterns = [
    # 调取网页
    path('admin/', admin.site.urls),
    path('index/', views.index),
    path('mulPage/', views.mulPage),
    path('mulUp/<str:model_name>', views.mulUp),
    path('dataset/<str:datasetName>', views.introduceDataset),
    # path('article/', views.article),
    path('article/<str:model_name>', views.article),
    path('model/<str:model_name>', views.model),
    path('layouts/', views.layouts),
    #path('back/',views.back_index),


    # 功能区
    path('output/<str:model_name>', views.output),
    path('mulOutput/', views.mulOutput),
    # upload File
    path('upload/<str:model_name>', views.upload),
    path('mulUpload/<str:model_name>', views.mulUpload),

    #进度条读取
    path('progress/', views.getProgress),
]
