import numpy as np
import pandas as pd
import warnings

from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from LogisticRegression import LogisticRegression
from msiProjekt.cross_validation_experiment import perform_cross_val
from msiProjekt.feature_selection_experiment import feature_selection, pca_feature_selection
from msiProjekt.t_test import t_test

# suppress warnings
warnings.filterwarnings('ignore')

# read data set
data = pd.read_csv('train.csv')

# dividing data into features (X) and labels (y)
X = data.drop(['Survived'], axis=1)
y = data['Survived']

# store column names, encode categorical variables, store column names encoded
col_names = X.columns.tolist()
X = pd.get_dummies(X, prefix_sep='_')
col_names_encoded = X.columns.tolist()

# scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# fill missing values with mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# initialize n_splits and n_repeats
n_splits = 2
n_repeats = 5
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# feature selection
X_selected = feature_selection(X, y, col_names_encoded, rkf)
X_selected_PCA = pca_feature_selection(X, y, col_names_encoded, rkf)

# find and return the best number of iterations, save results to file
#best_n_iter = find_best_n_iter(X_selected, rkf, y)

# create a classifier
lr = LogisticRegression(lr=0.001, n_iters=100)

# list of classifiers
classifiers = {'LR': lr,
               'kNN': KNeighborsClassifier(),
               'DTC': DecisionTreeClassifier(),
               'SVM': svm.SVC()
               }

# cross validation experiment
scores = perform_cross_val(classifiers, rkf, X_selected, y)

# train and predict with logistic regression on full data
lr.fit(X_selected, y)
y_predictions = lr.predict(X_selected)

# save predictions to file
np.save('predictions.npy', y_predictions)

# unclassified data plot
plt.scatter(data['Age'], y, color="b", marker="o", s=30)
plt.title("Niesklasyfikowane dane")
plt.xlabel("Age")
plt.ylabel("Survived")
plt.show()

# classified data plot
colors = np.array(['r', 'g'])  # green -> survived
plt.scatter(data['Age'], y_predictions, c=colors[y_predictions], s=30)
plt.title("Sklasyfikowane dane")
plt.xlabel("Age")
plt.ylabel("Survived")
plt.show()

# print classification report and confusion matrix
print(confusion_matrix(y, y_predictions))
print(classification_report(y, y_predictions))

# load saved files and analyze results
selected_features = np.load('selected_features.npy')
selected_features_pca = np.load('selected_features_pca.npy')
cross_validation = np.load('cross_validation_scores.npy')
predictions = np.load('predictions.npy')
n_iters_experiment = np.load('number_of_iterations_results.npz', allow_pickle=True)

# print selected features
print("Selected features:")
print(selected_features)
print("Selected PCA:")
print(selected_features_pca)

# print number of iters experiment results
# get keys of the saved data
n_iters_keys = n_iters_experiment.files

# print results for each key
for key in n_iters_keys:
    print(f"Results for {key}: ", n_iters_experiment[key])

t_test(cross_validation)
