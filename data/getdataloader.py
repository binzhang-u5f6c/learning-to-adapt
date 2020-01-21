import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Hyperplane1A(Dataset):
    def __init__(self):
        super(Hyperplane1A).__init__()
        df = pd.read_csv('data/hyperplane.1a.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_hyperplane1a(batch_size):
    ds = Hyperplane1A()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Hyperplane1B(Dataset):
    def __init__(self):
        super(Hyperplane1B).__init__()
        df = pd.read_csv('data/hyperplane.1b.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_hyperplane1b(batch_size):
    ds = Hyperplane1B()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Sea1A(Dataset):
    def __init__(self):
        super(Sea1A).__init__()
        df = pd.read_csv('data/sea.1a.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_sea1a(batch_size):
    ds = Sea1A()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Sea1B(Dataset):
    def __init__(self):
        super(Sea1A).__init__()
        df = pd.read_csv('data/sea.1b.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_sea1b(batch_size):
    ds = Sea1B()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Hyperplane2A(Dataset):
    def __init__(self):
        super(Hyperplane2A).__init__()
        df = pd.read_csv('data/hyperplane.2a.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_hyperplane2a(batch_size):
    ds = Hyperplane2A()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Hyperplane2B(Dataset):
    def __init__(self):
        super(Hyperplane2B).__init__()
        df = pd.read_csv('data/hyperplane.2b.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_hyperplane2b(batch_size):
    ds = Hyperplane2B()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Sea2A(Dataset):
    def __init__(self):
        super(Sea2A).__init__()
        df = pd.read_csv('data/sea.2a.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_sea2a(batch_size):
    ds = Sea2A()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl


class Sea2B(Dataset):
    def __init__(self):
        super(Sea2B).__init__()
        df = pd.read_csv('data/sea.2b.csv')
        self.x = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]

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


def get_sea2b(batch_size):
    ds = Sea2B()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl
