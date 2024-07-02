import h5py

with h5py.File('../data/combined/handlebar_gyro_train.h5', 'r') as hf:
    windows = hf['windows'][:]
    labels = hf['labels'][:]

labels = labels.astype(str)

print(windows.shape)

num_windows_to_display = 5
for i in range(num_windows_to_display):
    print(f"Window {i}:\n{windows[i]}")
    print(f"Label: {labels[i]}")
    print("\n\n")
