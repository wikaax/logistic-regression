import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


def perform_cross_val(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv=cv)

    # save cross-validation results to file
    np.save('cross_validation_scores.npy', scores)
    return scores
