import cv2
import numpy as np
import os

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.inception_v3 import preprocess_input
from keras.applications import imagenet_utils

import pdb

preprocess = imagenet_utils.preprocess_input

num_classes = 5

def load_mydata(img_rows, img_cols):
    inputShape = (img_rows, img_cols)
    X = []
    Y = []

    f = open("category.txt",'r') 
    file = f.readlines()
    f.close()
    categories = []
    for i in file:
        categories.append(i.strip())


    count = 0
    i = 0
    for category in categories:
        for root, dirs, files in os.walk(category):
            for file in files:
                if file.endswith(".jpg"):
                    try:                  
                        # load the input image using the Keras helper utility while ensuring
                        # the image is resized to `inputShape`, the required input dimensions
                        # for the ImageNet pre-trained network
                        
                        image = load_img(root + '/' + file, target_size=inputShape)
                        image = img_to_array(image)

                        image = image.transpose(2,1,0)
                        
                        # our input image is now represented as a NumPy array of shape
                        # (3, inputShape[0], inputShape[1]) however we need to expand the
                        # dimension by making the shape (1, 3, inputShape[0], inputShape[1])
                        # so we can pass it through thenetwork
                        image = np.expand_dims(image, axis=0)
                        
                        # pre-process the image using the appropriate function based on the
                        # model that has been loaded (i.e., mean subtraction, scaling, etc.)
                        image = preprocess(image)

                        X.append(image)
                        Y.append(i)
                        count += 1;
                        print(count, 'pictures done.')
                    except:
                        pass
        i += 1
 
    X = np.asarray(X).reshape((count, 3, img_rows, img_cols))
    Y = np.asarray(Y).reshape((count, 1))
    Y = np_utils.to_categorical(Y, num_classes)

    #create test and train samples indice
    temp = np.arange(count)
    np.random.shuffle(temp)

    train_sample_indice = temp[:int(len(temp) *0.8)]
    valid_sample_indice = temp[int(len(temp) *0.8):]

    X_train = X[train_sample_indice,:,:,:]
    Y_train = Y[train_sample_indice,:]

    X_valid = X[valid_sample_indice,:,:,:]
    Y_valid = Y[valid_sample_indice,:]
    
    return X_train, Y_train, X_valid, Y_valid