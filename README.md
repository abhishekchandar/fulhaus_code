# Image classification of furnitures using ResNet deep learning model

## Tools and technologies used:
- ``ResNet``: Transfer learning is performed in which the last layer is unfrozen to change the num of outputs to 3 as opposed to the default value of 1000 since ResNet is trained to predict 1000 labels. Since the classification involved Chair, Sofa and Bed, ResNet was an interesting choice as the model is already pretrained on a vast number of household items. Since the dataset size is relaively smaller, data augmentation was performed and the smaller version of ResNet was employed to avoid overfitting. 

- ``Django``: Backend framework using Python to develop an API wrapper for the ResNet model. Django provides mulitple advantanges such as enhanced security, sql database, framework for unit testing etc. out of the box. These are some of the reasons for choosing Django over Flask.

- ``multipart encoding`` of POST request to handle Image: The image from the client is sent to the server through multipart encoding as it allows to handle images with large file sizes unlike Base64 encoding which resulted in lenghty text representation of images with file size <1MB.

- ``Docker``: A lightway virtualization tool for easy reproducibity. The Django server is hosted in a Docker container which is able to self sustain on any platform without any dependency issues. Thanks to the Dockerfile - the recipe for building Docker images. 

- ``Github Actions`` for CI/CD: Employed CI/CD pipelines to test the code quality and the logic behind the functions developed. 

## Instructions to interact with the classifier as an API
- Clone the Github repository by running the command ``git clone https://github.com/abhishekchandar/fulhaus_code.git``
- Open the local terminal as admin and navigate to the root directory of the project i.e. ``BASE_URL/fulhaus_code/`` 
- Ensure that the Docker engine is running. In Windows, launch ``Docker Desktop`` application as admin to start the Docker daemon. 
- Build the Docker image using the Dockerfile by executing the following command: ``docker build . -t djangoapp``. NOTE, this command is executed in the directory where Dockerfile exists.
- Once the above build process is complete, run the docker container by executing the following command: ``docker run -it -p 8000:8000 djangoapp``. 
- The Django server is now up and running. 

## Instructions to send an image for inference as a POST request to the server
- Assumption: The Django server is up and running inside the Docker container which was instantiated from the previous set of instructions.
- Open Postman tool
- Set the request type to ``POST``, request URL to ``http://127.0.0.1:8000/api/classify``. Here, /api/classify is the API endpoint.
- Click the ``Body`` tab and set the type to ``form-data``.
- Set the ``KEY`` value to ``image``. On the right side of the ``KEY`` cell, set the type of ``VALUE`` to ``File`` by interacting with a dropdown menu.
- Set the ``VALUE`` by uploading the image to be classified. 
- Click on ``Send`` button for the POST API to reach the Django server. Note that the response is a JSON with two keys namely "status" and "data". The ``status`` key indicates if the POST request was successful. The ``data`` key holds the value of the predicted label.





