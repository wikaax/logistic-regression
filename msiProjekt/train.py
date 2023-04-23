import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# supress warnings
warnings.filterwarnings('ignore')

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker ="o", s = 30)
plt.show()

clf = LogisticRegression(lr=0.01)       # Classifier
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)

def accuracy(y_predictions, y_test):
    return np.sum(y_predictions==y_test)/len(y_test)

acc = accuracy(y_predictions, y_test)

print(acc)