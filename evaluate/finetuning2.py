import torch
import torch.nn.functional as F

from model.baselearner import BaseLearner
from utils.load import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def finetuning2(filename, batch_size1, batch_size2, hidden_size,
                meta_hidden_size, training_size, T, p):
    dataloader, input_size, output_size = get_dataloader(filename, batch_size1,
                                                         training_size, False)

    model = BaseLearner(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename[5:]+'.pt'))
    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False
    model.double()
    model.to(device)

    total = 0
    correct = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_x = batch_x.double()
        batch_x = batch_x.view(-1, batch_size2, input_size)
        batch_y = batch_y.to(device)
        batch_y = batch_y.long()
        batch_y = batch_y.view(-1, batch_size2)

        for j in range(batch_x.size(0)):
            ybar = model(batch_x[j])
            ybar = ybar.max(1)[1]
            ybar = ybar.view(-1)
            total += batch_size2
            correct += (ybar == batch_y[j]).sum().item()
            for t in range(T):
                model.zero_grad()
                ybar = model(batch_x[j])
                loss = F.nll_loss(ybar, batch_y[j])
                loss.backward()
                for para in model.parameters():
                    if para.requires_grad:
                        para.data.sub_(para.grad.data * 0.01)

    return correct, total
