from django.urls import path
from .views import *
from rest_framework import routers

urlpatterns = [
    path('api/classify', UploadView.as_view(), name = 'prediction')
    # path('api/upload/ct', CTUploadView.as_view(), name = 'ct_prediction'),
]
