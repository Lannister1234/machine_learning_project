# USAGE
# python read_features.py --model vgg16

import numpy as np
import argparse
from sklearn.svm import SVC
from collections import Counter
import pickle

ap = argparse.ArgumentParser()
ap.add_argument(
    "-model",
    "--model",
    type=str,
    default="vgg16",
    help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "vgg16": "_vgg16",
    "vgg19": "_vgg19",
    "resnet": "_resnet",
    "fine_tune_resnet": "_fine_tune_resnet"
}

filename = "feature_and_labels" + MODELS[args["model"]] + ".txt"

f = open(filename, 'r')
labels = []
features = []
filenames = []
for line in f.readlines():
    line = line.strip().split()
    labels.append(line[-2])
    filenames.append(line[-1])
    temp = [float(i) for i in line[:-2]]
    features.append(temp)
f.close()

try:
    f = open("indice.txt", 'r')
    all_samples = pickle.load(open("indice.txt", "rb"))
    f.close()
except:
    f = open("indice.txt", 'w')
    categories = {}
    for label in np.unique(labels):
        categories[label] = []

    for index, value in enumerate(labels):
        categories[value].append(index)

    min_category_num = Counter(labels).most_common()[-1][
        1]  #minimum samples numbers
    all_samples = []
    for label in np.unique(labels):
        np.random.shuffle(categories[label])
        all_samples.extend(categories[label][:min_category_num])

    np.random.shuffle(all_samples)

    pickle.dump(all_samples, open("indice.txt", "wb"))
    f.close()

#create test and train samples indice
train_sample_indice = all_samples[:int(len(all_samples) * 0.8)]
test_sample_indice = all_samples[int(len(all_samples) * 0.8):]

X_samples = np.asarray(features)  # n_samples * n_features
y_samples = np.asarray(labels)  # n_samples * 1

X_train = X_samples[train_sample_indice, :]
y_train = []
for i in train_sample_indice:
    y_train.append(y_samples[i])
y_train = np.asarray(y_train)

# can change the multiclass classifier
print("training ..")
print("sample nums", len(y_train))
clf = SVC()
clf.fit(X_train, y_train)

# parameters can change

SVC(C=1.0,
    cache_size=200,
    class_weight=None,
    coef0=0.0,
    decision_function_shape='ovr',
    degree=3,
    gamma='auto',
    kernel='rbf',
    max_iter=-1,
    probability=False,
    random_state=None,
    shrinking=True,
    tol=0.001,
    verbose=False)

X_pred = X_samples[test_sample_indice, :]
y_true = []
for i in test_sample_indice:
    y_true.append(y_samples[i])
y_true = np.asarray(y_true)

print("classifying ...")
y_pred = clf.predict(X_pred)

label_to_index = {
    "Peaceful": 0,
    "Passion": 1,
    "Fear": 2,
    "Happiness": 3,
    "Sadness": 4,
}

acc = 0.0
correct_num_class = [0 for x in range(5)]
sum_class = [0 for x in range(5)]
sum_predict_class = [0 for x in range(5)]
misclassfied_files = {}
for i in range(len(y_pred)):
    predict_class = label_to_index[y_pred[i]]
    true_class = label_to_index[y_true[i]]
    sum_predict_class[predict_class] += 1
    sum_class[true_class] += 1

    if (y_pred[i] == y_true[i]):
        correct_num_class[predict_class] += 1
        acc += 1
    else:
        fileList = misclassfied_files.get(y_true[i], list())
        fileList.append(filenames[i])
        misclassfied_files[y_true[i]] = fileList

print("Peaceful Passion Fear Happiness Sadness")
print("correct in each class: ")
print(correct_num_class)
print('\n')
print("total number in each class(truth)")
print(sum_class)
print("total accuracy:", acc / len(y_pred))
print('\n')
print("total number in each class(prediction)")
print(sum_predict_class)
precision = []
for i, j in zip(correct_num_class, sum_predict_class):
    precision.append(i / j)
print("precision of each class")
print(precision)

# print("misclassied files")
# print(misclassfied_files)
print('----------------------------')