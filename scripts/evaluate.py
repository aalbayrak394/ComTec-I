""" Evaluate different feature selection algorithms with RandomForestClssifier """
import sys
sys.path.append('..')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from feature_selection.fisher_score import fisher_score
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, RFE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score, f1_score

from utils.preprocessing import PreprocessingPipeline
from utils.feature_extraction import compute_features


# Load dataset
print('Loading dataset...')
ntp_intervals = {
    '1_marco': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
    '2_svenja': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
    '3_konstantin': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
    '4_aleyna': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
}
pipeline = PreprocessingPipeline(ntp_intervals)
splits = pipeline.run()

scores = {
    'baseline': [],
    'fisher': [],
    'pca': [],
    'knn': [],
    'svm': [],
}

print('# Evaluate feature selection algorithms with RandomForestClassifier.')
for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
    # Extract features
    print(f'### Extracting features from split {i+1}')
    X_train_features = compute_features(X_train)
    y_train = y_train.astype(str)

    X_test_features = compute_features(X_test)
    y_test = y_test.astype(str)

    # Train baseline model with all features
    rf = RandomForestClassifier(n_estimators=500, max_depth=70, criterion='log_loss')
    rf.fit(X_train_features, y_train)
    y_pred = rf.predict(X_test_features)
    baseline_score = f1_score(y_test, y_pred, average='micro')
    scores['baseline'].append(baseline_score)
    print(f'Baseline score: {baseline_score:.4f}')

    # # TODO: apply different feature selection algorithms
    # 1. Fisher Score
    fisher_selector = SelectKBest(fisher_score, k=10)
    X_train_fisher = fisher_selector.fit_transform(X_train_features, y_train)
    X_test_fisher = fisher_selector.transform(X_test_features)

    # 2. PCA
    pca = KernelPCA(n_components=10, kernel='sigmoid')
    X_train_pca = pca.fit_transform(X_train_features)
    X_test_pca = pca.transform(X_test_features)
    
    # 3. Sequential Feature Selector with KNN
    seq_selector = SequentialFeatureSelector(KNeighborsClassifier(), n_features_to_select=10)
    seq_selector.fit(X_train_features, y_train)
    X_train_knn = seq_selector.transform(X_train_features)
    X_test_knn = seq_selector.transform(X_test_features)

    # TODO: 4. Autoencoder  

    # TODO: 5. SVC
    svc = SVC(kernel="linear", C=1)
    svm_selector = RFE(estimator=svc, n_features_to_select=10, step=1)
    svm_selector.fit(X_train_features, y_train)
    X_train_svm = svm_selector.transform(X_train_features)
    X_test_svm = svm_selector.transform(X_test_features)

    # TODO: Evaluate each subset of features with RandomForestClassifier
    selected_features = {
        'fisher': (X_train_fisher, X_test_fisher),
        'pca': (X_train_pca, X_test_pca),
        'knn': (X_train_knn, X_test_knn),
        'svm': (X_train_svm, X_test_svm),
    }

    for name, (X_train_selected, X_test_selected) in selected_features.items():
        rf = RandomForestClassifier(n_estimators=500, max_depth=70, criterion='log_loss')
        rf.fit(X_train_selected, y_train)
        y_pred = rf.predict(X_test_selected)
        score = f1_score(y_test, y_pred, average='micro')
        scores[name].append(score)
        print(f'- {name}:  {score:.4f}')

# Print CV results
print('### Cross-validation results')
for name, values in scores.items():
    mean_score = sum(values) / len(values)
    print(f'- {name}:  {mean_score:.4f}')