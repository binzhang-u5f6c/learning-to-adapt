import torch.nn as nn
import torch.nn.functional as F


class BaseLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseLearner, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc1.bias)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.softmax(x, 1)
        return x
