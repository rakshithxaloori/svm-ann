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
    svm_tasks(X_train, y_train, X_test, y_test)
    ann_tasks(X_train, y_train, X_test, y_test)
