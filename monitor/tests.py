from django.test import TestCase
import os.path
import requests

url = "http://127.0.0.1:8000/api/classify"
script_dir = os.path.dirname(os.path.abspath(__file__))
class CreateTestForAPI(TestCase):
    def test_upload_request(self):
            form = {
                "image": open(os.path.join(script_dir,"Abner PU Swivel Chair.jpg"),'rb')
            }
            response = requests.post(url,files=form)
            self.assertEqual(response.status_code, 201)

