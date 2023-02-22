import urllib
from django.shortcuts import render
import numpy as np
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from PIL import Image
from torchvision import transforms

#Handling POST request from client
class UploadView(APIView):
    parser_classes = [JSONParser,MultiPartParser]
    @staticmethod
    def post(request):
        if request.method == 'POST':
            #Using request.FILES to handle file transfer through POST request
            files = request.FILES.get('image')
            #Defining ResNet model's object
            resnet = ResNetModelConfig.model
            #Opening image using pillow
            input_image = Image.open(files)
            #Basic preprocessing
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                resnet.to('cuda')

            with torch.no_grad():
                output = resnet(input_batch)
            # Tensor of shape 3, with confidence scores
            print(output[0])
            # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            print(probabilities) #Bed, Chair, Sofa
            # Read the categories
            categories = ["Bed","Chair","Sofa"]
            # Show top categories per image
            top5_prob, top5_catid = torch.topk(probabilities, 3)
            for i in range(top5_prob.size(0)):
                print(categories[top5_catid[i]], top5_prob[i].item()) #return categories[top5_catid[i]]

        return Response({
            'status': 'ok',
            'data': categories[top5_catid[0]],
        }, status=201)
