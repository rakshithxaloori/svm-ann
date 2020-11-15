import sys
import csv
from sklearn.model_selection import train_test_split

from classifier import svm_tasks, ann_tasks

def convert_row_to_int(row):
    new_row = []
    for value in row:
        new_row.append(float(value))

    return new_row

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python runner.py csv_path")

    csv_path = sys.argv[1]

    X = []
    y = []

    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        for row in reader:
            X.append(convert_row_to_int(row[:-1]))
            y.append(row[-1])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Perform tasks
    print("SVMs")
    clf_types = ['linear', 'poly', 'rbf']
    # C_values obtained emphirically
    C_values = [10.0, 100.0, 100.0]
    for C_value, clf_type in zip(C_values, clf_types):
        svm_tasks(X_train, y_train, X_test, y_test, C_value, clf_type)

    
    print("**********************************************************")
    print("**********************************************************")

    print("ANNs")
    # hidden layers, nodes, learning rate
    for hidden_layers in [(), (2,), (6,), (2, 3,), (3, 2,)]:
        for learning_rate in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            ann_tasks(X_train, y_train, X_test, y_test, hidden_layers, learning_rate)
