import numpy as np
import pandas as pd
import warnings

from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from msiProjekt.experiments.feature_selection_experiment import feature_selection
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

# # train and predict with logistic regression on full data
lr.fit(X_selected, y)
y_predictions = lr.predict(X_selected)

# RESULTS ANALYSIS
# load saved files and analyze results
selected_features = np.load('selected_features.npy')
cross_validation = np.load('cross_validation_scores.npy')
predictions = np.load('predictions.npy')
n_iters_experiment = np.load('number_of_iterations_results.npz')

# feature importance
feature_weights = lr.weights
feature_importance = np.abs(feature_weights)
features = ['Pclass', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Ticket_110152', 'Ticket_113760', 'Ticket_13502',
            'Ticket_24160', 'Ticket_2666', 'Ticket_29106', 'Ticket_347742', 'Ticket_CA. 2343', 'Ticket_PC 17572',
            'Ticket_PC 17755', 'Cabin_B96 B98', 'Cabin_E101', 'Cabin_F33', 'Embarked_C', 'Embarked_S']
importance = [0.016031662428189018, 0.0037305908758575074, 0.01203896354791156, 0.02567313154935162,
              0.02567313154935162, 0.003468582182947117, 0.003987848670741845, 0.0034700750345102696,
              0.0034445361913083387, 0.004016649766817327, 0.003540189598150869, 0.003530647252465814,
              0.0033744402951988006, 0.0034955961743442206, 0.003418423154435658, 0.003987848670741845,
              0.003491489266030324, 0.0034955100005501258, 0.007862900947430819, 0.007280306582351379]

# bar chart
plt.figure(figsize=(10, 6))
plt.bar(features, importance)
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# pie chart
plt.pie(importance, labels=features, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Ważność cech')
plt.show()

# print classification report and confusion matrix
print(confusion_matrix(y, predictions))
print(classification_report(y, predictions))

# confusion matrix plot
conf_mat = confusion_matrix(y, predictions)
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks([0, 1], ['Expected: 0', 'Expected: 1'])
plt.yticks([0, 1], ['Predicted: 0', 'Predicted: 1'])
plt.show()

# precision
precision_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
precision_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])

# recall
recall_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
recall_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])

precision0, recall0, thresholds = precision_recall_curve(y, predictions)
precision1, recall1, thresholds = precision_recall_curve(y, predictions)

# Pclass histogram
grouped_data = data.groupby(['Pclass', 'Survived']).size().unstack()
fig, ax = plt.subplots()
bar_width = 0.4
index = np.arange(len(grouped_data.index))
rects1 = ax.bar(index, grouped_data[0], bar_width, label="Didn't survive")
rects2 = ax.bar(index + bar_width, grouped_data[1], bar_width, label='Survived')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(grouped_data.index)
ax.set_ylabel('Number of people')
ax.set_title('Dependence of survival on passenger class (Pclass)')
ax.legend()
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y, predictions)
auc = roc_auc_score(y, predictions)
plt.plot(fpr, tpr, label=f'LR (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend()
plt.show()

# print selected features
print("Selected features:")
print(selected_features)

# print number of iters experiment results
n_iters_keys = n_iters_experiment
for key in n_iters_keys:
    print(f"Results for {key}: %.3f" % n_iters_experiment[key])

t_test(cross_validation)
