from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mode, skew
import scipy.signal as signal

# 3-axis accelerometer, 3-axis gyroscope -> 6 sensors
raw_features = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

def compute_features(window_dataset):
    engineered_features = []

    for window in tqdm(window_dataset):
        features = dict()

        for idx, sensor in enumerate(raw_features):
            # 1) Mean, 2) Median, 3) Mode
            features[f'mean_{sensor}'] = np.mean(window[:, idx])
            features[f'median_{sensor}'] = np.median(window[:, idx])
            features[f'mode_{sensor}'] = mode(window[:, idx])[0]

            # 4) Standard Deviation, 5) Variance
            features[f'std_{sensor}'] = np.std(window[:, idx])
            features[f'var_{sensor}'] = np.var(window[:, idx])

            # 6) Root Mean Square
            features[f'rms_{sensor}'] = np.sqrt(np.mean(window[:, idx] ** 2))

            # 7) Median Absolute Deviation
            features[f'mad_{sensor}'] = np.median(np.abs(window[:, idx] - np.median(window[:, idx])))

            # 8) Averaged Derivates
            features[f'avg_deriv_{sensor}'] = np.mean(np.diff(window[:, idx]))

            # 9) Skewness
            features[f'skew_{sensor}'] = skew(window[:, idx])

            # 10) Zero Crossing Rate
            features[f'zcr_{sensor}'] = ((window[:-1, idx] * window[1:, idx]) < 0).sum()

            # 11) Mean Crossing Rate
            mean = np.mean(window[:, idx])
            features[f'mcr_{sensor}'] = ((window[:-1, idx] - mean) * (window[1:, idx] - mean) < 0).sum()

            # 12) Time between peaks
            peaks, _ = find_peaks(window[:, idx])
            features[f'peak_time_{sensor}'] = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0

            # 13) Range
            features[f'range_{sensor}'] = np.ptp(window[:, idx])

            # 14) Interquartile Range
            features[f'iqr_{sensor}'] = np.percentile(window[:, idx], 75) - np.percentile(window[:, idx], 25)

            # 15) Max, 16) Min
            features[f'max_{sensor}'] = np.max(window[:, idx])
            features[f'min_{sensor}'] = np.min(window[:, idx])

            # 17) Spectral Entropy
            f, psd = signal.welch(window[:, idx], fs=50, nperseg=100)
            psd_norm = psd / np.sum(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features[f'spectral_entropy_{sensor}'] = spectral_entropy



        # Covariance
        for i in range(6):
            for j in range(i + 1, 6):
                if not f'cov_{raw_features[j]}_{raw_features[i]}' in features:
                    features[f'cov_{raw_features[i]}_{raw_features[j]}'] = np.cov(window[:, i], window[:, j])[0, 1]

        # Pairwise Correlation
        for i in range(6):
            for j in range(i + 1, 6):
                if not f'corr_{raw_features[j]}_{raw_features[i]}' in features:
                    features[f'corr_{raw_features[i]}_{raw_features[j]}'] = np.corrcoef(window[:, i], window[:, j])[0, 1]
    
        engineered_features.append(features)

    return pd.DataFrame(engineered_features)
