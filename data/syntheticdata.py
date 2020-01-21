from random import randint
from random import uniform

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import skmultiflow.data as mfdata


class SyntheticDataset(Dataset):
    def __init__(self, gtype, n, drift=True):
        super(SyntheticDataset).__init__()
        self.gtype = gtype
        self.n = n
        self.drift = drift
        if gtype == 'agrawal':
            self.generator = mfdata.AGRAWALGenerator(random_state=0)
        elif gtype == 'hyperplane1':
            self.generator = mfdata.HyperplaneGenerator(random_state=0)
        elif gtype == 'hyperplane2':
            self.generator = mfdata.HyperplaneGenerator(random_state=0)
        elif gtype == 'sea':
            self.generator = mfdata.SEAGenerator(random_state=0)
        elif gtype == 'sine':
            self.generator = mfdata.SineGenerator(random_state=0)
        self.generator.prepare_for_use()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        batch_size = len(idx)
        if self.drift:
            if self.gtype == 'agrawal':
                rd = randint(0, 9)
                self.generator.set_params(classification_function=rd)

            elif self.gtype == 'hyperplane1':
                rd = uniform(0, 0.05)
                self.generator.set_params(mag_change=rd)

            elif self.gtype == 'hyperplane2':
                rd = uniform(0, 1)
                self.generator.set_params(mag_change=rd)
                self.generator.next_sample()
                self.generator.set_params(mag_change=0)

            elif self.gtype == 'sea' or self.gtype == 'sine':
                rd = randint(0, 3)
                self.generator.set_params(classification_function=rd)

        x_ft, y_ft = self.generator.next_sample(batch_size)
        if self.gtype == 'hyperplane1':
            self.generator.set_params(mag_change=0)
        x_eval, y_eval = self.generator.next_sample(batch_size)

        x_ft = torch.from_numpy(x_ft)
        y_ft = torch.from_numpy(y_ft)
        x_eval = torch.from_numpy(x_eval)
        y_eval = torch.from_numpy(y_eval)
        return x_ft, y_ft, x_eval, y_eval


def get_synthetic_dataloader(gtype, n, batch_size, drift):
    ds = SyntheticDataset(gtype, n, drift)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=8, pin_memory=True)
    return dl
