import numpy as np
from sklearn.svm import SVC

X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])    # n_samples * n_features
y_train = np.array([1, 1, 2, 2])   # n_samples * 1


# can change the multiclass classifier
clf = SVC()
clf.fit(X_train, y_train)

# parameters can change

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

X_pred = np.array([[-0.8,-1],[1,1]])

print("classifying ...")
y_pred = clf.predict(X_pred)
print(y_pred)