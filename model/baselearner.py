import torch.nn as nn
import torch.nn.functional as F


class BaseLearner1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseLearner1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, 1)
        return x


class BaseLearner2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BaseLearner2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(hidden_size[1], hidden_size[2])
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        self.fc4 = nn.Linear(hidden_size[2], hidden_size[3])
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        self.fc5 = nn.Linear(hidden_size[3], output_size)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.softmax(x, 1)
        return x
