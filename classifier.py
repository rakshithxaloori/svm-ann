import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def test_classifier(clf, X, y):
    """ RETURN the accuracy on data X, y using classifier clf. """
    prediction_values = clf.predict(X)
    acc_count = 0
    for prediction, gold in zip(prediction_values, y):
        if prediction == gold:
            acc_count += 1

    return acc_count/len(y)


def svm_tasks(X_train, y_train, X_test, y_test):
    type_clf = ['linear', 'poly', 'rbf']
    clfs = []
    for each_type in type_clf:
        if each_type == 'poly':
            clfs.append(make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=each_type, degree=2)))
        else:
            clfs.append(make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=each_type)))

    np_X_train = np.array(X_train)
    np_y_train = np.array(y_train)

    # Testing
    print("SVMs")
    for each_type, each_clf in zip(type_clf, clfs):
        each_clf.fit(np_X_train, np_y_train)
        print("Accuracy with", each_type, "kernel:", test_classifier(each_clf, X_train, y_train), "on TRAIN data")
        print("Accuracy with", each_type, "kernel:", test_classifier(each_clf, X_test, y_test), "on TEST data")
        
def ann_tasks(X_train, y_train, X_test, y_test):
    pass
