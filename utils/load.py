import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.arff2pandas import arff2pandas


class DatasetFoo(Dataset):
    def __init__(self, filename, training_size, train=True):
        super(DatasetFoo).__init__()
        df = arff2pandas(filename)

        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

        self.x = pd.get_dummies(self.x)
        self.feature_num = len(self.x.keys())

        cols = self.y.keys()
        ybar = self.y.pop(cols[0])
        classes = pd.unique(ybar)
        self.class_num = len(classes)
        replace_map = {k: i for i, k in enumerate(classes)}
        self.y[cols[0]] = ybar.replace(replace_map)

        if train:
            self.x = self.x.iloc[:training_size, :]
            self.y = self.y.iloc[:training_size, :]
        else:
            self.x = self.x.iloc[training_size:, :]
            self.y = self.y.iloc[training_size:, :]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x.iloc[idx, :]
        x = x.to_numpy()
        x = torch.from_numpy(x)

        y = self.y.iloc[idx, :]
        y = y.to_numpy()
        y = torch.from_numpy(y)

        return x, y


def get_dataloader(filename, batch_size, training_size, train=True):
    ds = DatasetFoo(filename, training_size, train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl, ds.feature_num, ds.class_num
