from yaml import safe_load
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.arff2pandas import arff2pandas

# hyperparameters
with open('config.yml', 'r') as f:
    conf = safe_load(f)
split_point = conf['training_size']
batch_size = conf['batch_size']
num_workers = conf['num_workers']


class DatasetFoo(Dataset):
    def __init__(self, filename, train=True):
        super(DatasetFoo).__init__()
        df = arff2pandas(filename)

        if train:
            self.x = df.iloc[:split_point, :-1]
            self.y = df.iloc[:split_point, -1:]
        else:
            self.x = df.iloc[split_point:, :-1]
            self.y = df.iloc[split_point:, -1:]

        self.x = pd.get_dummies(self.x)
        self.feature_num = len(self.x.keys())

        cols = self.y.keys()
        ybar = self.y.pop(cols[0])
        classes = pd.unique(ybar)
        self.class_num = len(classes)
        replace_map = {k: i for i, k in enumerate(classes)}
        self.y[cols[0]] = ybar.replace(replace_map)

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


def get_dataloader(filename, train=True):
    ds = DatasetFoo(filename, train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return dl, ds.feature_num, ds.class_num
