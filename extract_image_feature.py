# USAGE
# python classify_image.py --model vgg16

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import Model
import numpy as np
import argparse
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
# print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

intermediate_model = Model(inputs=model.input,outputs=model.layers[-2].output)

f = open("category.txt",'r') #
file = f.readlines()
f.close()
categories = []
for i in file:
    categories.append(i.strip())
    
    
f = open("feature_and_labels.txt", 'w')   
cmd = "python extract_image_feature.py --image %s --model %s"
for category in categories:
    for root, dirs, files in os.walk(category):
        count = 0
        for file in files:
            if file.endswith(".jpg"):
                try:                  
                    # load the input image using the Keras helper utility while ensuring
                    # the image is resized to `inputShape`, the required input dimensions
                    # for the ImageNet pre-trained network
                    
                    image = load_img(root + '/' + file, target_size=inputShape)
                    image = img_to_array(image)
                    
                    # our input image is now represented as a NumPy array of shape
                    # (inputShape[0], inputShape[1], 3) however we need to expand the
                    # dimension by making the shape (1, inputShape[0], inputShape[1], 3)
                    # so we can pass it through thenetwork
                    image = np.expand_dims(image, axis=0)
                    
                    # pre-process the image using the appropriate function based on the
                    # model that has been loaded (i.e., mean subtraction, scaling, etc.)
                    image = preprocess(image)
                    
                    # extract features of RESNET-50
                    intermediate_feature = intermediate_model.predict(image)
                    
                    for i in intermediate_feature:
                        for feature in i:
                            f.write(str(feature) + ' ')                  
                    f.write(category + ' ' + root + '/' + file + '\n')
                    count += 1
                    print(count, 'pictures done.')
                except:
                    pass
        print(root + ' done!')
f.close()

# classify the image
#print("[INFO] classifying image with '{}'...".format(args["model"]))
# preds = model.predict(image)
# P = imagenet_utils.decode_predictions(preds)

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
# 	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
# orig = cv2.imread(args["image"])
# (imagenetID, label, prob) = P[0][0]
# cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)
