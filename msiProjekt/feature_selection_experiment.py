import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA


def pca_feature_selection(X, y, col_names, rkf):
    # list of n_features to find the best
    n_features_list = [5, 10, 15, 20, 25, 30]

    # storing results of n_features experiment
    results = {}
    best_n_feature = 0
    best_accuracy = 0

    for n_features in n_features_list:
        # apply PCA
        pca = PCA(n_components=n_features)
        X_selected = pca.fit_transform(X)

        # create logistic regression model
        lr = LogisticRegression()

        # perform cross validation
        scores = cross_val_score(lr, X_selected, y, cv=rkf)

        # store results
        results[n_features] = scores.mean()

        # update best results
        if scores.mean() > best_accuracy:
            best_n_feature = n_features
            best_accuracy = scores.mean()

    # apply PCA with best number of features
    pca = PCA(n_components=best_n_feature)
    X_selected = pca.fit_transform(X)

    # retrieve selected feature names
    selected_features = [f"PCA_{i + 1}" for i in range(best_n_feature)]

    # save selected features to file
    np.save('selected_features_pca.npy', selected_features)

    return X_selected


def feature_selection(X, y, col_names, rkf):
    # list of n_features to find the best
    n_features_list = [5, 10, 15, 20]

    # storing results of n_features experiment
    results = {}
    best_n_feature = 0
    best_accuracy = 0

    for n_features in n_features_list:
        # select n_features
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)

        # create logistic regression model
        lr = LogisticRegression()

        # perform cross validation
        scores = cross_val_score(lr, X_selected, y, cv=rkf)

        # store results
        results[n_features] = scores.mean()

        # update best results
        if scores.mean() > best_accuracy:
            best_n_feature = n_features
            best_accuracy = scores.mean()

    # select best number of features
    selector = SelectKBest(score_func=f_classif, k=best_n_feature)
    X_selected = selector.fit_transform(X, y)

    # retrieve selected feature names
    selected_features = np.array(col_names)[selector.get_support()]

    # save selected features to file
    np.save('selected_features.npy', selected_features)

    for n_features, accuracy in results.items():
        print(f"Number of Features: {n_features}, Accuracy: %.3f" % accuracy)
    return X_selected
