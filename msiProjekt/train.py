import numpy as np
import pandas as pd
import warnings

from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from msiProjekt.experiments.feature_selection_experiment import feature_selection, pca_feature_selection
from msiProjekt.experiments.iteration_experiment import find_best_n_iter
from msiProjekt.methods.cross_validation_method import perform_cross_val
from msiProjekt.methods.logistic_regression_method import LogisticRegression
from msiProjekt.methods.t_test_method import t_test

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
# best_n_iter = find_best_n_iter(X_selected, rkf, y)

# create a classifier
lr = LogisticRegression(lr=0.001, n_iters=100)

# list of classifiers
classifiers = {'LR': lr,
               'kNN': KNeighborsClassifier(),
               'DTC': DecisionTreeClassifier(),
               'SVM': svm.SVC(),
               'GNB': GaussianNB()
               }

# cross validation experiment
scores = perform_cross_val(classifiers, rkf, X_selected, y)

# train and predict with logistic regression on full data
lr.fit(X_selected, y)
y_predictions = lr.predict(X_selected)

# RESULTS ANALYSIS
# load saved files and analyze results
selected_features = np.load('selected_features.npy')
selected_features_pca = np.load('selected_features_pca.npy')
cross_validation = np.load('cross_validation_scores.npy')
predictions = np.load('predictions.npy')
# predictions = np.load('y_pred_LR.npy')
n_iters_experiment = np.load('number_of_iterations_results.npz', allow_pickle=True)

# print classification report and confusion matrix
print(confusion_matrix(y, predictions))
print(classification_report(y, predictions))

# confusion matrix plot
conf_mat = confusion_matrix(y, y_predictions)
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks([0, 1], ['Expected: 0', 'Expected: 1'])
plt.yticks([0, 1], ['Predicted: 0', 'Predicted: 1'])
plt.show()

# Group by Pclass and Survived
grouped_data = data.groupby(['Pclass', 'Survived']).size().unstack()
# Create a histogram
fig, ax = plt.subplots()
bar_width = 0.4
index = np.arange(len(grouped_data.index))
rects1 = ax.bar(index, grouped_data[0], bar_width, label="Didn't survive")
rects2 = ax.bar(index + bar_width, grouped_data[1], bar_width, label='Survived')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(grouped_data.index)
ax.set_ylabel('Number of people')
ax.set_title('Dependence of survival on ticket class (Pclass)')
ax.legend()
plt.show()

# ROC
fpr, tpr, thresholds = roc_curve(y, y_predictions)
auc = roc_auc_score(y, y_predictions)
plt.plot(fpr, tpr, label=f'LR (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Linia referencyjna
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend()
plt.show()

# print selected features
print("Selected features:")
print(selected_features)
print("Selected PCA:")
print(selected_features_pca)

# print number of iters experiment results
n_iters_keys = n_iters_experiment
for key in n_iters_keys:
    print(f"Results for {key}: %.3f" % n_iters_experiment[key])

t_test(cross_validation)
