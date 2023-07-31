################################################################
# TELESCOPE PARK CLASSIFICATION MODULE
# 2023 By BorisE
# based on https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb
#
# Test given images
# Used for testing. Recommend using park_webserber.py or write another using park_class_module.py
################################################################

#%env CUDA_VISIBLE_DEVICES=2

import torch, torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

from datetime import datetime
start_time = datetime.now()

from park_init import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("Using device: " + str(device))

dataset = 'dataset_park'

# Load the Data
# Set train and valid directory paths
train_directory = os.path.join(dataset, 'train')

 # Load Data from folders
data = {
    'train': datasets.ImageFolder(root=train_directory),
}

# Get a mapping of the indices to the class names, in order to see the output classes of the test images.
idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
print(idx_to_class)


################################################################
# Inference
def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    #print('Predict start. Duration: {}'.format(datetime.now() - start_time))
    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    #print('Predict test_image_tensor. Duration: {}'.format(datetime.now() - start_time))

    with torch.no_grad():
        model.eval()
        #print('Predict model.eval(). Duration: {}'.format(datetime.now() - start_time))

        # Model outputs log probabilities
        out = model(test_image_tensor)
        #print('Predict model(test_image_tensor). Duration: {}'.format(datetime.now() - start_time))
        ps = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)
        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]

        print("*** Image: " + test_image_name + " ***") 
        print("The most probable it is: " + cls + " [" + str(score) + "]") 
        #for i in range(2):
        #    print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
        print()


################################################################
# Test a particular model on a test image

model = torch.load(dataset + "/" + "{}_model_best.pt".format(dataset))
print('Duration: {}'.format(datetime.now() - start_time))

predict(model, 'AstroCam_1.jpg')
print('Duration: {}'.format(datetime.now() - start_time))

predict(model, 'AstroCam_2.jpg')
print('Duration: {}'.format(datetime.now() - start_time))

predict(model, 'AstroCam_3.jpg')

predict(model, 'AstroCam_4.jpg')

predict(model, 'AstroCam_5.jpg')

predict(model, 'AstroCam_6.jpg')

predict(model, 'AstroCam_7.jpg')

predict(model, 'AstroCam_8.jpg')

predict(model, 'AstroCam_9.jpg')

predict(model, 'AstroCam_10.jpg')

print('Duration: {}'.format(datetime.now() - start_time))

