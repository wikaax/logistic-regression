import numpy as np
from sklearn.metrics import accuracy_score


def perform_cross_val(classifiers, rkf, X, y):
    acc_scores = np.zeros(shape=[len(classifiers), rkf.get_n_splits()])

    for i, (train, test) in enumerate(rkf.split(X, y)):
        for j, (clf_name, clf) in enumerate(classifiers.items()):

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[j, i] = accuracy_score(y[test], y_pred)

    mean_scores = np.mean(acc_scores, axis=1)
    std_scores = np.std(acc_scores, axis=1)
    for clf_id, clf_name in enumerate(classifiers):
        print(list(classifiers.keys())[clf_id] + ":",
              "Mean score: %.3f" % mean_scores[clf_id],
              "\tStd: %.3f" % std_scores[clf_id])

    # save cross-validation results to file
    np.save('cross_validation_scores.npy', acc_scores)
    return acc_scores