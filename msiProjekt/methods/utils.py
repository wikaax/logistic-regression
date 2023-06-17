import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, classification_report


def confusion_matrix_and_classification_report(cross_val):
    conf_mat = cross_val[1]
    y_pred = cross_val[2]
    y_test = cross_val[3]

    # print classification report and confusion matrix
    print(conf_mat)

    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.xticks([0, 1], ['Expected: 0', 'Expected: 1'])
    plt.yticks([0, 1], ['Predicted: 0', 'Predicted: 1'])
    plt.show()

    print(classification_report(y_test, y_pred))

def roc_curve_plot(conf_mat, y_pred, y_test):
    # precision
    precision_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])
    precision_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])

    # recall
    recall_0 = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])
    recall_1 = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])

    precision0, recall0, thresholds = precision_recall_curve(y_pred, y_test)
    precision1, recall1, thresholds = precision_recall_curve(y_pred, y_test)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_pred, y_test)
    auc = roc_auc_score(y_pred, y_test)
    plt.plot(fpr, tpr, label=f'LR (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Reference line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()

def feature_selection_charts(lr):
    selected_features = np.load('../selected_features.npy')
    print(selected_features)

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

    # Pie chart with color map
    colors = plt.cm.tab20b(range(len(features)))
    plt.figure(figsize=(8, 8))
    plt.pie(importance, labels=features, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.axis('equal')
    plt.title('Ważność cech')
    plt.show()

def histograms(data):
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

    # Sex histogram
    grouped_data = data.groupby(['Sex', 'Survived']).size().unstack()
    fig, ax = plt.subplots()
    bar_width = 0.4
    index = np.arange(len(grouped_data.index))
    rects1 = ax.bar(index, grouped_data[0], bar_width, label="Didn't survive")
    rects2 = ax.bar(index + bar_width, grouped_data[1], bar_width, label='Survived')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(grouped_data.index)
    ax.set_ylabel('Number of people')
    ax.set_title('Dependence of survival on passenger sex')
    ax.legend()
    plt.show()

def feature_reduction_and_scatter_plot(col_names_encoded, X_selected):
    # dimension reduction using PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_selected)

    # feature weights for pc1
    feature_weights_pc1 = pca.components_[0]

    # feature weights for pc2
    feature_weights_pc2 = pca.components_[1]

    # pc1 feature importace
    top_features_pc1 = np.argsort(np.abs(feature_weights_pc1))[::-1][:5]
    print("Top features for PC1:")
    for feature_idx in top_features_pc1:
        print(col_names_encoded[feature_idx])

    # pc2 feature importace
    top_features_pc2 = np.argsort(np.abs(feature_weights_pc2))[::-1][:5]
    print("Top features for PC2:")
    for feature_idx in top_features_pc2:
        print(col_names_encoded[feature_idx])

    plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Scatter Plot after PCA')
    plt.show()

def iter_experiment_results():
    n_iters_experiment = np.load('../number_of_iterations_results.npz')

    # print number of iters experiment results
    n_iters_keys = n_iters_experiment
    for key in n_iters_keys:
        print(f"Results for {key}: %.3f" % n_iters_experiment[key])