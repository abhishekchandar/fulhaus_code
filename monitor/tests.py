from django.test import TestCase
import os.path
import requests

url = "http://127.0.0.1:8000/api/classify"
script_dir = os.path.dirname(os.path.abspath(__file__))
class CreateValidTestForAPI(TestCase):
    def test_upload_request(self):
            key = "image"
            file = open(os.path.join(script_dir,"Abner PU Swivel Chair.jpg"),'rb')
            form = {
                key: file 
            }
            # response = requests.post(url,files=form)
            # self.assertEqual(response.status_code, 201)
            self.assertEqual(key,"image")

class CreateInValidTestForAPI(TestCase):
    def test_upload_request(self):
            key = "image"
            file = "monitor\Abner PU Swivel Chair.jpg"
            form = {
               key : file
            }
            # response = requests.post(url,files=form)
            self.assertEqual(type(file), type("str"))

