import numpy as np
from sklearn.svm import SVC
from collections import Counter
import pdb

f = open("feature_and_labels1.txt", 'r')

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

categories = {}
for label in np.unique(labels):
    categories[label] = []

for index, value in enumerate(labels):
    categories[value].append(index)
    
min_category_num = Counter(labels).most_common()[-1][1] #minimum samples numbers
all_samples = []
for label in np.unique(labels):
    np.random.shuffle(categories[label])
    all_samples.extend(categories[label][:min_category_num])
    
#create test and train samples indice
np.random.shuffle(all_samples)
train_sample_indice = all_samples[:int(len(all_samples) *0.8)]
test_sample_indice = all_samples[int(len(all_samples) *0.8):]

X_samples = np.asarray(features)    # n_samples * n_features
y_samples = np.asarray(labels)   # n_samples * 1

X_train = X_samples[train_sample_indice,:]
y_train = []
for i in train_sample_indice:
	y_train.append(y_samples[i])
y_train = np.asarray(y_train)
print(Counter(y_train))

# can change the multiclass classifier
print("training ..")
print("sample nums", len(y_train))
clf = SVC()
clf.fit(X_train, y_train)

# parameters can change

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

X_pred = X_samples[test_sample_indice,:]
y_true = []
for i in test_sample_indice:
	y_true.append(y_samples[i])
y_true = np.asarray(y_true)
print(Counter(y_true))

print("classifying ...")
y_pred = clf.predict(X_pred)
print(y_pred)
print(y_train)
# pdb.set_trace()

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
misclassfied_files = {}
for i in range(len(y_pred)):
    predict_class = label_to_index[y_pred[i]]
    true_class = label_to_index[y_true[i]]
    sum_class[true_class] += 1

    if(y_pred[i]==y_true[i]):
        correct_num_class[predict_class] += 1
        acc += 1
    else:
        fileList = misclassfied_files.get(y_true[i], list())
        fileList.append(filenames[i])
        misclassfied_files[y_true[i]] = fileList

print("correct in each class: ")
print(correct_num_class)
print("total number in each class")
print(sum_class)
print("total accuracy:", acc/len(y_pred))
print("misclassied files")
print(misclassfied_files)