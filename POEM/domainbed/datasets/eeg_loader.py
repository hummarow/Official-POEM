import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EEGDataset(Dataset):
    def __init__(self, path: str):
        '''
        Create pytorch dataset from .h5 file.
        path: full path of the h5 file.
        '''
        f = h5py.File(path, "r")
        self.xs = []
        self.ys = []
        y = f["y"][:].squeeze()
        for x in f["x"][:]:
            self.xs.append(x.T)  # (63, 400) -> (400, 63)
            self.ys.append(y)

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)

        # Random shuffle
        idx = np.random.permutation(len(self.xs))
        self.xs, self.ys = self.xs[idx], self.ys[idx]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class EEGDataloader(DataLoader):
    def __init__(self, dataset: EEGDataset, batch_size=4, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataset(path: str):
    dataset = EEGDataset(path)
    return dataset
