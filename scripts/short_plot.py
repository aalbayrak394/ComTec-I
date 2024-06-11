import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_raw_data_csv():
    # df = pd.read_csv('../data/combined/backwheel_acc.csv')
    # df = pd.read_csv('../data/combined/backwheel_gyro.csv')
    df = pd.read_csv('../data/combined/handlebar_acc.csv')
    # df = pd.read_csv('../data/combined/handlebar_gyro.csv')
    df['NTP'] = pd.to_datetime(df['NTP'])

    plt.figure(figsize=(12, 8), dpi=500)
    plt.plot(df['NTP'], df['Acc-X'], label='Acc-X')
    plt.plot(df['NTP'], df['Acc-Y'], label='Acc-Y')
    plt.plot(df['NTP'], df['Acc-Z'], label='Acc-Z')
    # plt.plot(df['NTP'], df['Gyr-X'], label='Acc-X')
    # plt.plot(df['NTP'], df['Gyr-Y'], label='Acc-Y')
    # plt.plot(df['NTP'], df['Gyr-Z'], label='Acc-Z')

    plt.xlabel('Time (NTP)')
    plt.ylabel('Acc')
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=120))
    plt.grid()
    plt.show()


plot_raw_data_csv()
