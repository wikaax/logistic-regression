import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def perform_cross_val(classifiers, rkf, X, y):
    acc_scores = np.zeros(shape=[len(classifiers), rkf.get_n_splits()])
    conf_mat = None
    y_pred_final = None
    y_test_final = None

    for i, (train, test) in enumerate(rkf.split(X, y)):
        for j, (clf_name, clf) in enumerate(classifiers.items()):

            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            acc_scores[j, i] = accuracy_score(y[test], y_pred)

            if clf_name == 'LR':
                conf_mat = confusion_matrix(y[test], y_pred)
                y_pred_final = y_pred
                y_test_final = y[test]

                # Scatter plot
                # plt.figure(figsize=(6, 6))
                # plt.scatter(X[test][:, 0], X[test][:, 1], c=y_pred, cmap='viridis')
                # plt.title(f'Scatter Plot - Fold {i + 1}')
                # plt.xlabel('Feature 1')
                # plt.ylabel('Surivived')

    mean_scores = np.mean(acc_scores, axis=1)
    std_scores = np.std(acc_scores, axis=1)
    for clf_id, clf_name in enumerate(classifiers):
        print(list(classifiers.keys())[clf_id] + ":",
              "Mean score: %.3f" % mean_scores[clf_id],
              "\tStd: %.3f" % std_scores[clf_id])

    # save cross-validation score to file
    np.save('cross_validation_scores.npy', acc_scores)
    return acc_scores, conf_mat, y_pred_final, y_test_final
