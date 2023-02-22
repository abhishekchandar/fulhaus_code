from django.apps import AppConfig
from django.conf import settings
import os
import torch



class MonitorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'monitor'

#Class defining the object representing ResNet model
class ResNetModelConfig(AppConfig):
    name = 'resnetAPI'
    MODEL_FILE = os.path.join(settings.MODELS, "entire_model.pt")
    model = torch.load(MODEL_FILE)
