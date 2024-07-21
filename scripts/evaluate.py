""" Evaluate different feature selection algorithms with RandomForestClssifier """
import sys
sys.path.append('..')
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import accuracy_score, f1_score

from utils.preprocessing import PreprocessingPipeline
from utils.feature_extraction import compute_features


NTP_INTERVALS = {
    '1_marco': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
    '2_svenja': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
    '3_konstantin': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
    '4_aleyna': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
}

N_FEATURES_TO_SELECT = 30

# Load dataset
print('Loading dataset...')
pipeline = PreprocessingPipeline(NTP_INTERVALS)
splits = pipeline.create_cv_splits()

scores = {
    'baseline': [],
    'fisher': [],
    'pca': [],
    'knn': [],
    'svc': [], 
}

# load selected features from json file
feature_names = json.load(open(f'../results/selected_features_{N_FEATURES_TO_SELECT}.json', 'r'))

print('# Evaluate feature selection algorithms with RandomForestClassifier.')
for i, (X_train, y_train, X_test, y_test) in enumerate(splits):
    # Extract features
    print(f'### Extracting features from split {i+1}')
    X_train_features = compute_features(X_train)
    y_train = y_train.astype(str)

    X_test_features = compute_features(X_test)
    y_test = y_test.astype(str)

    # Train baseline model with all features
    rf = RandomForestClassifier(criterion='entropy', random_state=0)
    rf.fit(X_train_features, y_train)
    y_pred = rf.predict(X_test_features)
    baseline_score = f1_score(y_test, y_pred, average='micro')
    scores['baseline'].append(baseline_score)
    print(f'Baseline score: {baseline_score:.4f}')

    # Supervised feature selection (Fisher Score, SVC, KNN)
    X_train_fisher = X_train_features[feature_names['fisher']]
    X_test_fisher = X_test_features[feature_names['fisher']]

    X_train_svc = X_train_features[feature_names['svc']]
    X_test_svc = X_test_features[feature_names['svc']]

    X_train_knn = X_train_features[feature_names['knn']]
    X_test_knn = X_test_features[feature_names['knn']]

    # TODO: Unsupervised dimensionality reduction (PCA, Autoencoder)
    pca = PCA(n_components=N_FEATURES_TO_SELECT).fit(X_train_features)
    X_train_pca = pca.transform(X_train_features)
    X_test_pca = pca.transform(X_test_features)
    

    # TODO: Evaluate each subset of features with RandomForestClassifier
    selected_features = {
        'fisher': (X_train_fisher, X_test_fisher),
        'pca': (X_train_pca, X_test_pca),
        'knn': (X_train_knn, X_test_knn),
        'svc': (X_train_svc, X_test_svc),
    }

    for name, (X_train_selected, X_test_selected) in selected_features.items():
        rf = RandomForestClassifier(criterion='entropy', random_state=0)
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