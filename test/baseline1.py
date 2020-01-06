import torch

from model.baselearner import BaseLearner
from utils.load import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def baseline1(filename):
    dataloader, input_size, output_size = get_dataloader(filename, False)

    hidden_size = int((input_size + output_size) / 2)
    model = BaseLearner(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename[5:]+'.pt'))
    model.to(device)

    total = 0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_x = batch_x.double()
            batch_y = batch_y.to(device)
            batch_y = batch_y.long()
            batch_y = batch_y.view(-1)

            ybar = model(batch_x)
            ybar = ybar.max(1)[1]
            ybar = ybar.view(-1)
            total += batch_y.size(0)
            correct += (ybar == batch_y).sum().item()

    return correct, total
