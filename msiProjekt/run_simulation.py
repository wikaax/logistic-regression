import pandas as pd
import warnings

from sklearn import svm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from msiProjekt.experiments.feature_selection_experiment import feature_selection
from msiProjekt.experiments.iteration_experiment import find_best_n_iter
from msiProjekt.methods.cross_validation_method import perform_cross_val
from msiProjekt.methods.logistic_regression_method import LogisticRegression
from msiProjekt.methods.t_test_method import t_test
from msiProjekt.methods.utils import histograms, feature_reduction_and_scatter_plot, feature_selection_charts, \
    iter_experiment_results, confusion_matrix_and_classification_report, mean_scores, scatter, precision_and_recall, \
    roc_curve_plot

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

# create a classifier
lr = LogisticRegression(lr=0.001, n_iters=100)

# list of classifiers
classifiers = {'LR': lr,
               'kNN': KNeighborsClassifier(),
               'DTC': DecisionTreeClassifier(),
               'SVM': svm.SVC(),
               'GNB': GaussianNB()
               }

# initialize n_splits and n_repeats
n_splits = 2
n_repeats = 5
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# feature selection
X_selected = feature_selection(X, y, col_names_encoded, rkf, lr)

# find and return the best number of iterations, save results to file
# best_n_iter = find_best_n_iter(X_selected, rkf, y)

# cross validation results
cross_val = perform_cross_val(classifiers, rkf, X_selected, y)

# RESULTS ANALYSIS
precision_recall_scores = precision_and_recall(cross_val)
roc_curve_plot(cross_val)
# print('MEAN SCORES: ')
# mean_scores(classifiers, cross_val)
# print('ITER_EXPERIMENT RESULTS: ')
# iter_experiment_results()
# print('T_TEST RESULTS: ')
# t_test()
# print('CONFUSION MATRIX AND CLASSIFICATION REPORT: ')
# confusion_matrix_and_classification_report(cross_val)

# feature_selection_charts(lr)
feature_reduction_and_scatter_plot(cross_val)
# histograms(data)
