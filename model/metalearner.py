import torch.nn as nn
import torch.nn.functional as F


class MetaLearner(nn.Module):
    def __init__(self, hidden_size):
        super(MetaLearner, self).__init__()
        self.lstm1 = nn.LSTM(4, hidden_size, 1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, 2, 1, batch_first=True)

    def forward(self, x, hc1, hc2):
        x, hc1 = self.lstm1(x, hc1)
        x, hc2 = self.lstm2(x, hc2)
        x = F.softmax(x, 2)
        return x, hc1, hc2
