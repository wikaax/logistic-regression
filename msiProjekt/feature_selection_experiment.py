import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(X, y, col_names):

    # select 5 features
    selector = SelectKBest(score_func=f_classif, k=5)
    X_selected = selector.fit_transform(X, y)

    # retrieve selected feature names
    selected_features = np.array(col_names)[selector.get_support()]

    # save selected features to file
    np.save('selected_features.npy', selected_features)

    return X_selected