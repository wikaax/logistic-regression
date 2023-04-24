import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import figure
import seaborn as sns
from LogisticRegression import LogisticRegression

# supress warnings
warnings.filterwarnings('ignore')

# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target

data = pd.read_csv('train.csv')

# Dividing data into features (X) and labels (y)
X = data.drop(['Survived'], axis=1)
y = data['Survived']

X = pd.get_dummies(X, prefix_sep='_')
y = LabelEncoder().fit_transform(y)
X = StandardScaler().fit_transform(X)

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

print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))

def accuracy(y_predictions, y_test):
    return np.sum(y_predictions==y_test)/len(y_test)

acc = accuracy(y_predictions, y_test)

print(acc)