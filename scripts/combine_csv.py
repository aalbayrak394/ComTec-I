import numpy as np
import pandas as pd
import os


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
            nearest_data_index = (data['NTP'] - label_row['NTP']).abs().idxmin()
            col_name = 'Roughness_Label' if val == 'roughness_high' or val == 'roughness_medium' or val == 'roughness_low' else 'Curb_Label'
            data.at[nearest_data_index, col_name] = val
        return data

    if not os.path.exists(output_file_):
        with open(output_file_, 'w') as f:
            f.write('')

    for file_path, file_info in ntp_intervals_.items():
        ntp_start, ntp_end = file_info['interval']
        ntp_start = pd.to_datetime(ntp_start)
        ntp_end = pd.to_datetime(ntp_end)

        filtered_data = filter_and_select_columns(file_path, ntp_start, ntp_end)

        # Resample to 10Hz
        filtered_data.set_index('NTP', inplace=True, drop=False)
        filtered_data = filtered_data.resample('100ms').first()

        for label_type, label_file in file_info.items():
            if label_type.startswith('Label'):
                label_df = pd.read_csv(label_file)
                filtered_data = add_label_to_data(filtered_data, label_df)
        filtered_data['Roughness_Label'] = filtered_data['Roughness_Label'].replace('', np.nan)
        filtered_data['Roughness_Label'] = filtered_data['Roughness_Label'].ffill()

        if os.path.getsize(output_file_) > 0:
            filtered_data.to_csv(output_file_, mode='a', header=False, index=False)
        else:
            filtered_data.to_csv(output_file_, mode='w', header=True, index=False)


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
        'interval': ('2024-05-28 15:56:29.423', '2024-05-28 16:09:52.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label.0.csv',
        'LabelOther': '../data/label/3_konstantin/Label/Label.0.csv'
    },
    '../data/backwheel/4_aleyna/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/backwheel_acc.csv'
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
        'interval': ('2024-05-28 15:56:29.423', '2024-05-28 16:09:52.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label.0.csv',
        'LabelOther': '../data/label/3_konstantin/Label/Label.0.csv'
    },
    '../data/backwheel/4_aleyna/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/backwheel_gyro.csv'
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
        'interval': ('2024-05-28 15:56:29.423', '2024-05-28 16:09:52.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label.0.csv',
        'LabelOther': '../data/label/3_konstantin/Label/Label.0.csv'
    },
    '../data/handlebar/4_aleyna/Accelerometer/Accelerometer.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/handlebar_acc.csv'
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
        'interval': ('2024-05-28 15:56:29.423', '2024-05-28 16:09:52.000'),
        'LabelSelf': '../data/handlebar/3_konstantin/Label/Label.0.csv',
        'LabelOther': '../data/label/3_konstantin/Label/Label.0.csv'
    },
    '../data/handlebar/4_aleyna/Gyroscope/Gyroscope.0.csv': {
        'interval': ('2024-05-28 16:11:26.149', '2024-05-28 16:21:35.000'),
        'LabelSelf': '../data/handlebar/4_aleyna/Label/Label.0.csv',
        'LabelOther': '../data/label/4_aleyna/Label/Label.0.csv'
    },
}
output_file = '../data/combined/handlebar_gyro.csv'
combine_csv_files(ntp_intervals, output_file, "Gyr")

print("Finished")
