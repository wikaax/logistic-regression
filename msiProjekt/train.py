import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import figure
import seaborn as sns
from LogisticRegression import LogisticRegression

# supress warnings
warnings.filterwarnings('ignore')

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
X_new = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y,
    test_size = 0.2,
    random_state = 1234)
#
# fig = plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], y, color="b", marker ="o", s = 30)
# plt.show()

# selector = SelectKBest(score_func=f_classif, k=5)
# X_train_new = selector.fit_transform(X_train, y_train)
# X_test_new = selector.transform(X_test)

# train and predict with logistic regression
clf = LogisticRegression(lr=0.01)
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)

# calculate and print the accuracy
def accuracy(y_predictions, y_test):
    return np.sum(y_predictions==y_test)/len(y_test)

acc = accuracy(y_predictions, y_test)

print("Accuracy: ", acc)

# print classification report and confusion matrix
print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))