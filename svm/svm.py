import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mode, skew
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

with h5py.File('../data/combined/handlebar_me_acc_test.h5', 'r') as hf:
    windows_test = hf['windows'][:]
    labels_test = hf['labels'][:]

labels_test = labels_test.astype(str)

with h5py.File('../data/combined/handlebar_me_acc_train.h5', 'r') as hf:
    windows_train = hf['windows'][:]
    labels_train = hf['labels'][:]

labels_train = labels_train.astype(str)


def calculate_features(window):
    features = {}
    # Mean, Median, Mode
    features['mean_x'] = np.mean(window[:, 0])
    features['mean_y'] = np.mean(window[:, 1])
    features['mean_z'] = np.mean(window[:, 2])

    features['median_x'] = np.median(window[:, 0])
    features['median_y'] = np.median(window[:, 1])
    features['median_z'] = np.median(window[:, 2])

    features['mode_x'] = mode(window[:, 0])[0]
    features['mode_y'] = mode(window[:, 1])[0]
    features['mode_z'] = mode(window[:, 2])[0]

    # Standard Deviation, Variance
    features['std_x'] = np.std(window[:, 0])
    features['std_y'] = np.std(window[:, 1])
    features['std_z'] = np.std(window[:, 2])

    features['var_x'] = np.var(window[:, 0])
    features['var_y'] = np.var(window[:, 1])
    features['var_z'] = np.var(window[:, 2])

    # Covariance Matrix
    cov_matrix = np.cov(window, rowvar=False)
    features['cov_xy'] = cov_matrix[0, 1]
    features['cov_xz'] = cov_matrix[0, 2]
    features['cov_yz'] = cov_matrix[1, 2]

    # Root Mean Square
    features['rms_x'] = np.sqrt(np.mean(window[:, 0] ** 2))
    features['rms_y'] = np.sqrt(np.mean(window[:, 1] ** 2))
    features['rms_z'] = np.sqrt(np.mean(window[:, 2] ** 2))

    # Median Absolute Deviation
    features['mad_x'] = np.median(np.abs(window[:, 0] - np.median(window[:, 0])))
    features['mad_y'] = np.median(np.abs(window[:, 1] - np.median(window[:, 1])))
    features['mad_z'] = np.median(np.abs(window[:, 2] - np.median(window[:, 2])))

    # Averaged Derivates
    features['avg_deriv_x'] = np.mean(np.diff(window[:, 0]))
    features['avg_deriv_y'] = np.mean(np.diff(window[:, 1]))
    features['avg_deriv_z'] = np.mean(np.diff(window[:, 2]))

    # Skewness
    features['skew_x'] = skew(window[:, 0])
    features['skew_y'] = skew(window[:, 1])
    features['skew_z'] = skew(window[:, 2])

    # Zero Crossing Rate
    features['zcr_x'] = ((window[:-1, 0] * window[1:, 0]) < 0).sum()
    features['zcr_y'] = ((window[:-1, 1] * window[1:, 1]) < 0).sum()
    features['zcr_z'] = ((window[:-1, 2] * window[1:, 2]) < 0).sum()

    # Mean Crossing Rate
    mean_x = np.mean(window[:, 0])
    mean_y = np.mean(window[:, 1])
    mean_z = np.mean(window[:, 2])
    features['mcr_x'] = ((window[:-1, 0] - mean_x) * (window[1:, 0] - mean_x) < 0).sum()
    features['mcr_y'] = ((window[:-1, 1] - mean_y) * (window[1:, 1] - mean_y) < 0).sum()
    features['mcr_z'] = ((window[:-1, 2] - mean_z) * (window[1:, 2] - mean_z) < 0).sum()

    # Pairwise Correlation
    features['corr_xy'] = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
    features['corr_xz'] = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
    features['corr_yz'] = np.corrcoef(window[:, 1], window[:, 2])[0, 1]

    # Time between peaks
    peaks_x, _ = find_peaks(window[:, 0])
    peaks_y, _ = find_peaks(window[:, 1])
    peaks_z, _ = find_peaks(window[:, 2])
    features['peak_time_x'] = np.mean(np.diff(peaks_x)) if len(peaks_x) > 1 else 0
    features['peak_time_y'] = np.mean(np.diff(peaks_y)) if len(peaks_y) > 1 else 0
    features['peak_time_z'] = np.mean(np.diff(peaks_z)) if len(peaks_z) > 1 else 0

    # Range
    features['range_x'] = np.ptp(window[:, 0])
    features['range_y'] = np.ptp(window[:, 1])
    features['range_z'] = np.ptp(window[:, 2])

    # Interquartile Range
    features['iqr_x'] = np.percentile(window[:, 0], 75) - np.percentile(window[:, 0], 25)
    features['iqr_y'] = np.percentile(window[:, 1], 75) - np.percentile(window[:, 1], 25)
    features['iqr_z'] = np.percentile(window[:, 2], 75) - np.percentile(window[:, 2], 25)

    # Max, Min
    features['max_x'] = np.max(window[:, 0])
    features['max_y'] = np.max(window[:, 1])
    features['max_z'] = np.max(window[:, 2])

    features['min_x'] = np.min(window[:, 0])
    features['min_y'] = np.min(window[:, 1])
    features['min_z'] = np.min(window[:, 2])

    return features


features_train = []
for i in range(0, windows_train.shape[0]):
    feature_dict_train = calculate_features(windows_train[i])
    features_train.append(feature_dict_train)

features_test = []
for i in range(0, windows_test.shape[0]):
    feature_dict_test = calculate_features(windows_test[i])
    features_test.append(feature_dict_test)

feature_df_train = pd.DataFrame(features_train)
feature_df_test = pd.DataFrame(features_test)

svc = SVC(kernel="linear", C=1)
selector = RFE(estimator=svc, n_features_to_select=5, step=1)
selector = selector.fit(feature_df_train, labels_train)

selected_features = feature_df_train.columns[selector.support_]

print("Features:")
for feature in selected_features:
    print(feature)

X_test_selected = selector.transform(feature_df_test)
svc.fit(selector.transform(feature_df_train), labels_train)
accuracy = svc.score(X_test_selected, labels_test)
print("Accuracy:", accuracy)
