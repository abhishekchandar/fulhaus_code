from django.urls import path
from .views import *
from rest_framework import routers

#api/classify is the API endpoint for ResNet inference. 
urlpatterns = [
    path('api/classify', UploadView.as_view(), name = 'prediction')
    # path('api/upload/ct', CTUploadView.as_view(), name = 'ct_prediction'),
]
