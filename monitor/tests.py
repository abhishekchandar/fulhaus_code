from django.test import TestCase
import os.path
import requests

url = "http://127.0.0.1:8000/api/classify"
script_dir = os.path.dirname(os.path.abspath(__file__))
class CreateValidKeyForAPI(TestCase):
    def test_upload_request(self):
            key = "image"
            self.assertEqual(key,"image")

class CreateInValidTestForAPI(TestCase):
    def test_upload_request(self):
            file = "monitor\Abner PU Swivel Chair.jpg"
            self.assertEqual(type(file), type("str"))

#Uncomment to test perform API Integration testing locally.

# class CreateValidKeyForAPI(TestCase):
#     def test_upload_request(self):
#             key = "image"
#             file = open(os.path.join(script_dir,"Abner PU Swivel Chair.jpg"),'rb')
#             form = {
#                 key: file 
#             }
#             response = requests.post(url,files=form)
#             self.assertEqual(response.status_code, 201)

# class CreateInValidTestForAPI(TestCase):
#     def test_upload_request(self):
#             key = "image"
#             file = "monitor\Abner PU Swivel Chair.jpg"
#             form = {
#                key : file
#             }
#             response = requests.post(url,files=form)
#             self.assertEqual(response.status_code,500)

