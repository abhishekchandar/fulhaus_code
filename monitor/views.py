import urllib
from django.shortcuts import render
import numpy as np
from .apps import *
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, JSONParser
from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt
# import cv2


# Create your views here.
class UploadView(APIView):
    # parser_classes = (
    #     MultiPartParser,
    #     JSONParser,
    # )
    parser_classes = [JSONParser,MultiPartParser]
    @staticmethod
    def post(request):
        if request.method == 'POST':
            files = request.FILES.get('image')
            print(files)
            # def evaluate_model(model, test_batches):
            #   """
            #     Evaluate a trained model.
            #   """
            #   # Evaluate model
            #   score = model.evaluate(test_batches, verbose=0)
            #   print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
            #  evaluate_model(model_ft,)

            # sample execution (requires torchvision)
            # #load models
            resnet = ResNetModelConfig.model
            input_image = Image.open(files)
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
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
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

        # upload_data = cloudinary.uploader.upload(file)
        #print(upload_data)
        # img = upload_data['url']



        # req = urllib.request.urlopen(img)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # image = cv2.imdecode(arr, -1) # 'Load it as it is'
        # #image = cv2.imread('upload_chest.jpg') # read file 
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
        # image = cv2.resize(image,(224,224))
        # image = np.array(image) / 255
        # image = np.expand_dims(image, axis=0)

        # resnet_pred = resnet_chest.predict(image)
        # probability = resnet_pred[0]
        # #print("Resnet Predictions:")
        # if probability[0] > 0.5:
        #     resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        # else:
        #     resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        # #print(resnet_chest_pred)

        # vgg_pred = vgg_chest.predict(image)
        # probability = vgg_pred[0]
        # #print("VGG Predictions:")
        # if probability[0] > 0.5:
        #     vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        # else:
        #     vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        # #print(vgg_chest_pred)

        # inception_pred = inception_chest.predict(image)
        # probability = inception_pred[0]
        # #print("Inception Predictions:")
        # if probability[0] > 0.5:
        #     inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        # else:
        #     inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        # #print(inception_chest_pred)

        # xception_pred = xception_chest.predict(image)
        # probability = xception_pred[0]
        # #print("Xception Predictions:")
        # if probability[0] > 0.5:
        #     xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
        # else:
        #     xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
        #print(xception_chest_pred)
        return Response({
            'status': 'ok',
            'data': categories[top5_catid[0]],
        }, status=201)


# class CTUploadView(APIView):
#     parser_classes = (
#         MultiPartParser,
#         JSONParser,
#     )

#     @staticmethod
#     def post(request):
#         file = request.data.get('picture')
#         # upload_data = cloudinary.uploader.upload(file)
#         # #print(upload_data)
#         # img = upload_data['url']


#         # #load models
#         # resnet_chest = ResNetCTModelConfig.model
#         # vgg_chest = VGGCTModelConfig.model
#         # inception_chest = InceptionCTModelConfig.model
#         # xception_chest = ExceptionCTModelConfig.model

#         # req = urllib.request.urlopen(img)
#         # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#         # image = cv2.imdecode(arr, -1) # 'Load it as it is'
#         # #image = cv2.imread('upload_chest.jpg') # read file 
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
#         # image = cv2.resize(image,(224,224))
#         # image = np.array(image) / 255
#         # image = np.expand_dims(image, axis=0)

#         # resnet_pred = resnet_chest.predict(image)
#         # probability = resnet_pred[0]
#         # #print("Resnet Predictions:")
#         # if probability[0] > 0.5:
#         #     resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
#         # else:
#         #     resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
#         # #print(resnet_chest_pred)

#         # vgg_pred = vgg_chest.predict(image)
#         # probability = vgg_pred[0]
#         # #print("VGG Predictions:")
#         # if probability[0] > 0.5:
#         #     vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
#         # else:
#         #     vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
#         # #print(vgg_chest_pred)

#         # inception_pred = inception_chest.predict(image)
#         # probability = inception_pred[0]
#         # #print("Inception Predictions:")
#         # if probability[0] > 0.5:
#         #     inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
#         # else:
#         #     inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
#         # #print(inception_chest_pred)

#         # xception_pred = xception_chest.predict(image)
#         # probability = xception_pred[0]
#         # #print("Xception Predictions:")
#         # if probability[0] > 0.5:
#         #     xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
#         # else:
#         #     xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% NonCOVID')
#         #print(xception_chest_pred)
#         return Response({
#             'status': 'success',
#             'data': "hello",
#             # 'url':img,
#             # 'xceptionCT_chest_pred':xception_chest_pred,
#             # 'inceptionCT_chest_pred':inception_chest_pred,
#             # 'vggCT_chest_pred':vgg_chest_pred,
#             # 'resnetCT_chest_pred':resnet_chest_pred,
#         }, status=201)