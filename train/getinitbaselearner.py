import torch
import torch.nn.functional as F
import torch.optim as optim

from model.baselearner import BaseLearner
from utils.load import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_init_baselearner(filename, hidden_size, batch_size,
                         training_size, epoch, lr):
    dataloader, input_size, output_size = get_dataloader(filename,
                                                         batch_size,
                                                         training_size)

    model = BaseLearner(input_size, hidden_size, output_size)
    model.to(device)
    model.double()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for e in range(epoch):
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_x = batch_x.double()
            batch_y = batch_y.to(device)
            batch_y = batch_y.long()
            batch_y = batch_y.view(-1)

            optimizer.zero_grad()

            ybar = model(batch_x)
            loss = F.nll_loss(ybar, batch_y)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), filename[5:]+'.pt')
