import numpy as np
import pandas as pd
import warnings

import sklearn
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# suppress warnings
warnings.filterwarnings('ignore')

# read data set
data = pd.read_csv('train.csv')

# dividing data into features (X) and labels (y)
X = data.drop(['Survived'], axis=1)
y = data['Survived']

# encode categorical variables
X = pd.get_dummies(X, prefix_sep='_')

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# fill missing values with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# feature selection
selector = SelectKBest(score_func=f_classif, k=5)
X = selector.fit_transform(X, y)

# initialize n_splits and n_repeats
n_splits = 2
n_repeats = 5
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# create a classifier
clf = LogisticRegression(lr=0.0001, n_iters=1000)

# perform cross-validation
scores = cross_val_score(clf, X, y, cv=rkf)

# train and predict with logistic regression on full data
clf.fit(X, y)
y_predictions = clf.predict(X)

# unclassified data
plt.scatter(data['Age'], y, color="b", marker="o", s=30)
plt.title("Niesklasyfikowane dane")
plt.xlabel("Age")
plt.ylabel("Survived")
plt.show()

# classified data
colors = np.array(['r', 'g'])   # green -> survived
plt.scatter(data['Age'], y_predictions, c=colors[y_predictions], s=30)
plt.title("Sklasyfikowane dane")
plt.xlabel("Age")
plt.ylabel("Survived")
plt.show()

# calculate and print the accuracy
def accuracy(y_predictions, y_test):
    return np.sum(y_predictions==y_test)/len(y_test)

acc = accuracy(y_predictions, y)

print("Accuracy: ", acc)

# print classification report and confusion matrix
print(confusion_matrix(y, y_predictions))
print(classification_report(y, y_predictions))
