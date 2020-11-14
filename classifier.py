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


def svm_tasks(X_train, y_train, X_test, y_test, C_value, clf_type):
    if clf_type == 'poly':
        clf = make_pipeline(StandardScaler(), SVC(C=C_value, gamma='auto', kernel=clf_type, degree=2))
    else:
        clf = make_pipeline(StandardScaler(), SVC(C=C_value, gamma='auto', kernel=clf_type))

    np_X_train = np.array(X_train)
    np_y_train = np.array(y_train)

    # Testing
    clf.fit(np_X_train, np_y_train)
    print("kernel:", clf_type)
    print("C_value:", C_value)
    print("Accuracy:", test_classifier(clf, X_train, y_train), "on TRAIN data")
    print("Accuracy:", test_classifier(clf, X_test, y_test), "on TEST data")
    print("-----------------------------------------------------")
        
def ann_tasks(X_train, y_train, X_test, y_test):
    pass
