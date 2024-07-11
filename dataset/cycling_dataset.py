import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CyclingDataset(Dataset):
    def __init__(self, file_path_acc, file_path_gyro):
        with h5py.File(file_path_acc, 'r') as hf:
            data_acc = hf['windows'][:]
            labels_acc = hf['labels'][:]

        with h5py.File(file_path_gyro, 'r') as hf:
            data_gyro = hf['windows'][:]
            labels_gyro = hf['labels'][:]
        
        assert len(data_acc) == len(data_gyro)
        assert len(labels_acc) == len(labels_gyro)
        assert np.all(labels_acc == labels_gyro)

        self.data = np.concatenate([data_acc, data_gyro], axis=-1)
        self.labels = labels_acc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]