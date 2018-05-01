# USAGE
# python extract_image_feature.py --model vgg16

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.models import load_model
from keras.applications import Xception  # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import Model
import numpy as np
from PIL import Image
import argparse
from sklearn.svm import SVC
from collections import Counter
from sklearn.externals import joblib
import argparse
import os
import pickle

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
    "xception": Xception,  # TensorFlow ONLY
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

# Network = MODELS[args["model"]]
# model = Network()
# model.load_weights("resnet50_best.h5")
model = load_model('resnet50_best.h5')

intermediate_model = Model(inputs=model.input, outputs=model.layers[-2].output)

filename = "features_test_images" + ".txt"

f = open(filename, 'w')
cmd = "python extract_image_feature.py --image %s --model %s"
test_dir = "test_images"
for root, dirs, files in os.walk(test_dir):
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
                f.write(root + '/' + file + '\n')
                count += 1
                print(count, 'pictures done.')
            except:
                pass
    print(root + ' done!')
f.close()


# ap = argparse.ArgumentParser()
# ap.add_argument("-model", "--model", type=str, default="vgg16",
#                 help="name of pre-trained network to use")
# args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "vgg16": "_vgg16",
    "vgg19": "_vgg19",
    "resnet": "_resnet"
}

f = open(filename, 'r')

# labels = []
features = []
filenames = []
for line in f.readlines():
    line = line.strip().split()
    # labels.append(line[-2])
    filenames.append(line[-1])
    temp = [float(i) for i in line[:-1]]
    features.append(temp)
f.close()

X_samples = np.asarray(features)  # n_samples * n_features

# can change the multiclass classifier
print("predicting labels ..")
print("sample nums", len(X_samples))

#with open('SVM/clf.pkl', 'rb') as f:
clf = joblib.load('SVM/clf.pkl')
y_pred = clf.predict(X_samples)

print(y_pred)

classified_path = "classified_images"
for i in range(len(y_pred)):
    label = y_pred[i].decode("utf-8")
    fileName = filenames[i]
    print(label)
    if not os.path.exists(classified_path + "/" + label):
        os.makedirs(classified_path + "/" + label)
    try:
        img = Image.open(fileName)
        ind = fileName.index('/')
        fileName = fileName[ind + 1:]
        print(fileName)
        img.save(classified_path + "/" + label + "/" + fileName)
    except IOError:
        pass
    


# label_to_index = {
#     "Peaceful": 0,
#     "Passion": 1,
#     "Fear": 2,
#     "Happiness": 3,
#     "Sadness": 4,
# }
#
# acc = 0.0
# correct_num_class = [0 for x in range(5)]
# sum_class = [0 for x in range(5)]
# misclassfied_files = {}
# for i in range(len(y_pred)):
#     predict_class = label_to_index[y_pred[i]]
#     true_class = label_to_index[y_true[i]]
#     sum_class[true_class] += 1
#
#     if (y_pred[i] == y_true[i]):
#         correct_num_class[predict_class] += 1
#         acc += 1
#     else:
#         fileList = misclassfied_files.get(y_true[i], list())
#         fileList.append(filenames[i])
#         misclassfied_files[y_true[i]] = fileList
#
# print("correct in each class: ")
# print(correct_num_class)
# print("total number in each class")
# print(sum_class)
# print("total accuracy:", acc / len(y_pred))
# print("misclassied files")
# print(misclassfied_files)


