import numpy as np
from sklearn.svm import SVC


f = open("feature_and_labels.txt", 'r') 

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

#create test and train samples indice
all_samples_num = len(labels)
a = np.arange(all_samples_num)
temp = np.random.shuffle(a)

train_sample_indice = temp[:int(len(temp) *0.8)]
test_sample_indice = temp[int(len(temp) *0.8):]


X_samples = np.asarray(features)    # n_samples * n_features
y_samples = np.asarray(labels)   # n_samples * 1

X_train = X_samples[train_sample_indice,:]
y_train = []
for i in train_sample_indice:
	y_train.append(y_samples[i])
y_train = np.asarray(y_train)

# can change the multiclass classifier
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

print("classifying ...")
y_pred = clf.predict(X_pred)
# print(y_pred)
# print(y_train)

acc = np.sum(y_pred == y_true)
print("acc:",acc/y_pred)

