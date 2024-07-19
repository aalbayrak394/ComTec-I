import numpy as np
import pandas as pd

def fisher_score(X, y):
    X = np.array(X)
    y = np.array(y)
    classes = np.unique(y)
    print(classes)
    n_features = X.shape[1]
    
    fisher_scores = np.zeros(n_features)
    
    for i in range(n_features):
        feature = X[:, i]
        overall_mean = np.mean(feature)
        numerator = 0
        denominator = 0
        
        for c in classes:
            class_feature = feature[y == c]
            class_mean = np.mean(class_feature)
            class_variance = np.var(class_feature)
            class_size = len(class_feature)
            
            numerator += class_size * (class_mean - overall_mean) ** 2
            denominator += class_size * class_variance
            
        fisher_scores[i] = numerator / denominator
    
    return fisher_scores