from django.test import TestCase,Client
from rest_framework.test import APIRequestFactory
from rest_framework.response import Response
from rest_framework import status
import socket
import json
from PIL import Image
import os.path
import http.client
import requests

# conn = http.client.HTTPConnection("127.0.0.1",port=8000)
# headers = {"content_type": "multipart/form-data"}

url = "http://127.0.0.1:8000/api/classify"
script_dir = os.path.dirname(os.path.abspath(__file__))
# file = Image.open(os.path.join(script_dir,"Abner PU Swivel Chair.jpg"))
class CreateTestForAPI(TestCase):
    def test_upload_request(self):
            form = {
                "image": open(os.path.join(script_dir,"Abner PU Swivel Chair.jpg"),'rb')
            }
            response = requests.post(url,files=form)
            self.assertEqual(response.status_code, 201)