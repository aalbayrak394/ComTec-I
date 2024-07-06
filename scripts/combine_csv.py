import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def combine_csv_files(ntp_intervals_, output_file_, sensor_type):
    def filter_and_select_columns(file_path_, ntp_start_, ntp_end_):
        df = pd.read_csv(file_path_)
        df['NTP'] = pd.to_datetime(df['NTP'])
        filtered_df = df[(df['NTP'] > ntp_start_) & (df['NTP'] < ntp_end_)].copy()
        selected_columns = filtered_df[['NTP', f'{sensor_type}-X', f'{sensor_type}-Y', f'{sensor_type}-Z']].copy()
        selected_columns['Roughness_Label'] = ''
        selected_columns['Curb_Label'] = ''
        return selected_columns

    def add_label_to_data(data, label_df_):
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

    def create_sliding_windows_with_labels(label_rough, label_curb, window_size, step_size, scaled_data_):
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

    windows_train_together, windows_test_together, labels_train_together, labels_test_together = None, None, None, None

    for file_path, file_info in ntp_intervals_.items():
        ntp_start, ntp_end = file_info['interval']
        ntp_start = pd.to_datetime(ntp_start)
        ntp_end = pd.to_datetime(ntp_end)

        filtered_data = filter_and_select_columns(file_path, ntp_start, ntp_end)

        # Resample to 100Hz
        filtered_data.set_index('NTP', inplace=True, drop=False)
        filtered_data = filtered_data.resample('10ms').first()

        # Add Labels
        for label_type, label_file in file_info.items():
            if label_type.startswith('Label'):
                label_df = pd.read_csv(label_file)
                filtered_data = add_label_to_data(filtered_data, label_df)
        filtered_data['Roughness_Label'] = filtered_data['Roughness_Label'].replace('', np.nan)
        filtered_data['Roughness_Label'] = filtered_data['Roughness_Label'].ffill()

        # Fix for missing value for timeslot
        filtered_data.iloc[:, :5] = filtered_data.iloc[:, :5].ffill()
        filtered_data['Curb_Label'] = filtered_data['Curb_Label'].fillna('')

        # Test-Train
        windows_train, windows_test, labels_train, labels_test = train_test_split(
            filtered_data.iloc[:, 1:4], filtered_data.iloc[:, 4:6], test_size=0.2, shuffle=False
        )

        scaler = StandardScaler()
        scaler.fit(windows_train)
        train_scaled = scaler.transform(windows_train)
        test_scaled = scaler.transform(windows_test)

        # Sliding Windows
        windows_train_scaled, labels_train_scaled = create_sliding_windows_with_labels(labels_train["Roughness_Label"],
                                                                                       labels_train["Curb_Label"], 100,
                                                                                       50,
                                                                                       train_scaled)
        windows_test_scaled, labels_test_scaled = create_sliding_windows_with_labels(labels_test["Roughness_Label"],
                                                                                     labels_test["Curb_Label"], 100,
                                                                                     50,
                                                                                     test_scaled)

        if windows_train_together is None:
            windows_train_together = windows_train_scaled
            windows_test_together = windows_test_scaled
            labels_train_together = labels_train_scaled
            labels_test_together = labels_test_scaled
        else:
            windows_train_together = np.concatenate((windows_train_together, windows_train_scaled), axis=0)
            windows_test_together = np.concatenate((windows_test_together, windows_test_scaled), axis=0)
            labels_train_together = np.concatenate((labels_train_together, labels_train_scaled), axis=0)
            labels_test_together = np.concatenate((labels_test_together, labels_test_scaled), axis=0)

    with h5py.File(output_file_ + '_train.h5', 'w') as hf_train:
        hf_train.create_dataset('windows', data=windows_train_together)
        hf_train.create_dataset('labels', data=labels_train_together)

    with h5py.File(output_file_ + '_test.h5', 'w') as hf_test:
        hf_test.create_dataset('windows', data=windows_test_together)
        hf_test.create_dataset('labels', data=labels_test_together)


# TODO: auf die jeweils richtigen Label beschränken


ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/backwheel/3_konstantin/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
}
output_file = '../data/combined/backwheel_me_acc'
combine_csv_files(ntp_intervals, output_file, "Acc")

ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/handlebar/3_konstantin/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
}
output_file = '../data/combined/handlebar_me_acc'
combine_csv_files(ntp_intervals, output_file, "Acc")

ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/backwheel/3_konstantin/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
}
output_file = '../data/combined/backwheel_me_gyro'
combine_csv_files(ntp_intervals, output_file, "Gyr")

ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/handlebar/3_konstantin/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
}
output_file = '../data/combined/handlebar_me_gyro'
combine_csv_files(ntp_intervals, output_file, "Gyr")

# backwheel_acc
ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/backwheel/1_marco/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
        'LabelSelf': '../data/handlebar/1_marco/Label/Label.0.csv',
        'LabelOther': '../data/label/1_marco/Label/Label.0.csv'
    },
    '../data/backwheel/2_svenja/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
        'LabelSelf': '../data/handlebar/2_svenja/Label/Label.0.csv',
        'LabelOther': '../data/label/2_svenja/Label/Label.0.csv'
    },
    '../data/backwheel/3_konstantin/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
    '../data/backwheel/4_aleyna/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/backwheel_acc'
combine_csv_files(ntp_intervals, output_file, "Acc")

# backwheel_gyro
ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/backwheel/1_marco/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
        'LabelSelf': '../data/handlebar/1_marco/Label/Label.0.csv',
        'LabelOther': '../data/label/1_marco/Label/Label.0.csv'
    },
    '../data/backwheel/2_svenja/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
        'LabelSelf': '../data/handlebar/2_svenja/Label/Label.0.csv',
        'LabelOther': '../data/label/2_svenja/Label/Label.0.csv'
    },
    '../data/backwheel/3_konstantin/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
    '../data/backwheel/4_aleyna/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/backwheel_gyro'
combine_csv_files(ntp_intervals, output_file, "Gyr")

# handlebar_acc
ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/handlebar/1_marco/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
        'LabelSelf': '../data/handlebar/1_marco/Label/Label.0.csv',
        'LabelOther': '../data/label/1_marco/Label/Label.0.csv'
    },
    '../data/handlebar/2_svenja/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
        'LabelSelf': '../data/handlebar/2_svenja/Label/Label.0.csv',
        'LabelOther': '../data/label/2_svenja/Label/Label.0.csv'
    },
    '../data/handlebar/3_konstantin/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
    '../data/handlebar/4_aleyna/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/handlebar_acc'
combine_csv_files(ntp_intervals, output_file, "Acc")

# handlebar_gyro
ntp_intervals = {  # file+path: (start_ntp, end_ntp)
    '../data/handlebar/1_marco/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:21:46.830', '2024-05-28 15:36:21.000'),
        'LabelSelf': '../data/handlebar/1_marco/Label/Label.0.csv',
        'LabelOther': '../data/label/1_marco/Label/Label.0.csv'
    },
    '../data/handlebar/2_svenja/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:39:02.218', '2024-05-28 15:52:16.613'),
        'LabelSelf': '../data/handlebar/2_svenja/Label/Label.0.csv',
        'LabelOther': '../data/label/2_svenja/Label/Label.0.csv'
    },
    '../data/handlebar/3_konstantin/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 15:56:31.000', '2024-05-28 16:09:37.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label_edited.csv'
    },
    '../data/handlebar/4_aleyna/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/handlebar_gyro'
combine_csv_files(ntp_intervals, output_file, "Gyr")

print("Finished")
