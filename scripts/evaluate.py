""" Evaluate different feature selection algorithms with RandomForestClassifier """
import sys

from sklearn.svm import SVC

sys.path.append('..')

from sklearn.ensemble import RandomForestClassifier
from feature_selection.fisher_score import fisher_score
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score, f1_score

from dataset.cycling_dataset import get_dataset
from utils.feature_extraction import compute_features


# read segmented data
train_dataset = get_dataset(
    file_path_acc='../data/preprocessed/handlebar_acc_train.h5',
    file_path_gyro='../data/preprocessed/handlebar_gyro_train.h5',
)

test_dataset = get_dataset(
    file_path_acc='../data/preprocessed/handlebar_acc_test.h5',
    file_path_gyro='../data/preprocessed/handlebar_gyro_test.h5',
)

# extract features
windows_train, labels_train = train_dataset
X_train = compute_features(windows_train)
y_train = labels_train.astype(str)

windows_test, labels_test = test_dataset
X_test = compute_features(windows_test)
y_test = labels_test.astype(str)

# train baseline model with all features
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)
baseline_train_score = f1_score(y_train, y_pred_train, average='micro')
baseline_test_score = f1_score(y_test, y_pred_test, average='micro')

# TODO: apply different feature selection algorithms
# 1. Fisher Score
fisher_selector = SelectKBest(fisher_score, k=30)
X_train_fisher = fisher_selector.fit_transform(X_train, y_train)
X_test_fisher = fisher_selector.transform(X_test)

# 2. PCA
pca = KernelPCA(n_components=10, kernel='sigmoid')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# TODO: 3. KNN

# TODO: 4. Autoencoder

# TODO: 5. SVC

svc = SVC(kernel="linear", C=1)
svm_selector = RFE(estimator=svc, n_features_to_select=10, step=1)
X_train_svm = svm_selector.fit_transform(X_train, y_train)
X_test_svm = svm_selector.transform(X_test)

# TODO: evaluate each subset of features with RandomForestClassifier
selected_features = {
    'fisher': (X_train_fisher, X_test_fisher),
    'pca': (X_train_pca, X_test_pca),
    'svm': (X_train_svm, X_test_svm)
}

print(f'Baseline train score: {baseline_train_score:.2f}, test score: {baseline_test_score:.2f}')

for name, (X_train_selected, X_test_selected) in selected_features.items():
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train_selected, y_train)
    y_pred_train = rf.predict(X_train_selected)
    y_pred_test = rf.predict(X_test_selected)
    train_accuracy = f1_score(y_train, y_pred_train, average='micro')
    test_accuracy = f1_score(y_test, y_pred_test, average='micro')
    print(f'{name} train score: {train_accuracy:.2f}, test score: {test_accuracy:.2f}')