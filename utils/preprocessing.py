import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreprocessingPipeline:
    def __init__(self, ntp_intervals, output_file=None):
        self.ntp_intervals = ntp_intervals
        self.output_file = output_file

    def _filter_and_select_columns(self, file_path_, ntp_start_, ntp_end_, sensor_type):
        df = pd.read_csv(file_path_)
        df['NTP'] = pd.to_datetime(df['NTP'])
        filtered_df = df[(df['NTP'] > ntp_start_) & (df['NTP'] < ntp_end_)].copy()
        selected_columns = filtered_df[['NTP', f'{sensor_type}-X', f'{sensor_type}-Y', f'{sensor_type}-Z']].copy()
        selected_columns['Roughness_Label'] = ''
        selected_columns['Curb_Label'] = ''
        return selected_columns
    
    def _add_label_to_data(self, data, label_df_):
        label_df_['NTP'] = pd.to_datetime(label_df_['NTP'])
        for _, label_row in label_df_.iterrows():
            val = label_row['Label']
            if val == 'curb_none':
                continue
            if val == 'roughness_none':
                val = 'roughness_low'
            if val == 'curb_low' or val == 'curb_high':
                val = 'curb'
            if val == 'roughness_medium':
                val = 'roughness_high'
            nearest_data_index = (data['NTP'] - label_row['NTP']).abs().idxmin()
            col_name = 'Roughness_Label' if val == 'roughness_high' or val == 'roughness_medium' or val == 'roughness_low' else 'Curb_Label'
            data.at[nearest_data_index, col_name] = val
        return data
    
    def _create_sliding_windows_with_labels(self, label_rough, label_curb, window_size, step_size, scaled_data_):
        num_windows = (len(scaled_data_) - window_size) // step_size + 1
        windows_ = np.zeros((num_windows, window_size, scaled_data_.shape[1]))
        labels_ = []

        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            windows_[i] = scaled_data_[start_idx:end_idx]

            label_curb_window = label_curb.iloc[start_idx:end_idx]
            if (label_curb_window != '').any():
                window_label = label_curb_window[label_curb_window != ''].iloc[0]
            else:
                window_label = label_rough.iloc[start_idx:end_idx].mode()[0]

            labels_.append(window_label)

        return windows_, np.array(labels_, dtype='S')
    
    def run(self):
        persons = self.ntp_intervals.keys()
        train_sets = []

        for name in persons:
            # load person's data and concat accelerometer and gyroscope
            acc_file = f'../data/handlebar/{name}/Accelerometer/Accelerometer.0.csv'
            gyro_file = f'../data/handlebar/{name}/Gyroscope/Gyroscope.0.csv'
            acc_data = self._filter_and_select_columns(
                acc_file,
                self.ntp_intervals[name][0],
                self.ntp_intervals[name][1],
                'Acc'
            )
            gyro_data = self._filter_and_select_columns(
                gyro_file,
                self.ntp_intervals[name][0],
                self.ntp_intervals[name][1],
                'Gyr'
            )

            # Resample to 100Hz
            acc_data.set_index('NTP', inplace=True, drop=False)
            acc_data = acc_data.resample('10ms').first()

            gyro_data.set_index('NTP', inplace=True, drop=True)
            gyro_data = gyro_data.resample('10ms').first()

            gyro_data.drop(columns=['Roughness_Label', 'Curb_Label'], inplace=True)
            data = pd.concat([acc_data, gyro_data], axis=1, join='inner')

            # Add Labels
            label_df = pd.read_csv(f'../data/handlebar/{name}/Label/Label.0.csv')
            data = self._add_label_to_data(data, label_df)
            data['Roughness_Label'] = data['Roughness_Label'].replace('', np.nan)
            data['Roughness_Label'] = data['Roughness_Label'].ffill()

            columns_order = ['NTP', 'Acc-X', 'Acc-Y', 'Acc-Z', 'Gyr-X', 'Gyr-Y', 'Gyr-Z', 'Roughness_Label', 'Curb_Label']
            data = data.reindex(columns=columns_order)

            # Fix for missing value for timeslot
            data.iloc[:, :8] = data.iloc[:, :8].ffill()
            data['Curb_Label'] = data['Curb_Label'].fillna('')
            
            train_sets.append(data)

        # Create CV splits
        splits = []
        for set in train_sets:
            test_set = set
            train_set = pd.concat([x for x in train_sets if x is not set])

            # Fit scaler on train set and fit on train and test set
            X_train = train_set[['Acc-X', 'Acc-Y', 'Acc-Z', 'Gyr-X', 'Gyr-Y', 'Gyr-Z']]
            y_train = train_set[['Roughness_Label', 'Curb_Label']]
            X_test = test_set[['Acc-X', 'Acc-Y', 'Acc-Z', 'Gyr-X', 'Gyr-Y', 'Gyr-Z']]
            y_test = test_set[['Roughness_Label', 'Curb_Label']]

            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create sliding windows
            windows_train, labels_train = self._create_sliding_windows_with_labels(
                y_train['Roughness_Label'],
                y_train['Curb_Label'],
                window_size=100,
                step_size=50,
                scaled_data_=X_train_scaled
            )
            windows_test, labels_test = self._create_sliding_windows_with_labels(
                y_test['Roughness_Label'],
                y_test['Curb_Label'],
                window_size=100,
                step_size=50,
                scaled_data_=X_test_scaled
            )

            splits.append((windows_train, labels_train, windows_test, labels_test))

        return splits
