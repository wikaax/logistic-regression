import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def perform_cross_val(classifiers, rkf, X, y):
    acc_scores = np.zeros(shape=[len(classifiers), rkf.get_n_splits()])
    precision_scores = []
    recall_scores = []
    conf_mat = None
    y_pred_final = None
    y_test_final = None
    x_test = None
    x_train = None
    y_train = None

    for i, (train, test) in enumerate(rkf.split(X, y)):
        for j, (clf_name, clf) in enumerate(classifiers.items()):

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[j, i] = accuracy_score(y[test], y_pred)

            if clf_name == 'LR':
                conf_mat = confusion_matrix(y[test], y_pred)

                precision_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
                precision_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
                recall_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
                recall_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
                precision_scores.append((precision_0, precision_1))
                recall_scores.append((recall_0, recall_1))

                y_pred_final= y_pred
                y_test_final = y[test]
                x_test = X[test]
                x_train = X[train]
                y_train = y[train]

    # save cross-validation score to file
    np.save('cross_validation_scores.npy', acc_scores)
    return acc_scores, conf_mat, y_pred_final, y_test_final, x_test, x_train, y_train, precision_scores, recall_scores
