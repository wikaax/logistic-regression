import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
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
X_new = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_new,
    y,
    test_size = 0.5,
    random_state = 1234)

# train and predict with logistic regression
clf = LogisticRegression(lr=0.01)
# y_predictions = cross_val_predict(clf, X_new, y, cv=2)
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)

# unclassified data
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
plt.title("Niesklasyfikowane dane")
plt.xlabel("X")
plt.ylabel("Survived")
plt.show()

# classified data
plt.scatter(X_test[:, 0], y_predictions, color="b", marker="o", s=30)
plt.title("Sklasyfikowane dane")
plt.xlabel("X")
plt.ylabel("Survived")
plt.show()

# calculate and print the accuracy
def accuracy(y_predictions, y_test):
    return np.sum(y_predictions==y_test)/len(y_test)

acc = accuracy(y_predictions, y_test)

print("Accuracy: ", acc)

# print classification report and confusion matrix
print(confusion_matrix(y_test, y_predictions))
print(classification_report(y_test, y_predictions))