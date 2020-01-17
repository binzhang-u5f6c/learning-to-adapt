import torch.nn as nn


class MetaLearner(nn.Module):
    def __init__(self, hidden_size):
        super(MetaLearner, self).__init__()
        self.lstm = nn.LSTM(4, hidden_size, 2, batch_first=True)
        nn.init.orthogonal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_ih_l1)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l1)
        nn.init.ones_(self.lstm.bias_ih_l0)
        nn.init.ones_(self.lstm.bias_ih_l1)
        nn.init.ones_(self.lstm.bias_hh_l0)
        nn.init.ones_(self.lstm.bias_hh_l1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, hc):
        x, hc = self.lstm(x, hc)
        x = self.fc(x)
        return x, hc
