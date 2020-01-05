import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaLearner(nn.Module):
    def __init__(self, hidden_size):
        super(MetaLearner, self).__init__()
        self.lstm = nn.LSTM(4, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, hc):
        x, hc = self.lstm(x, hc)
        x = self.fc(x)
        x = F.softmax(x, 2)
        return x, hc
