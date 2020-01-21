import torch
import torch.nn.functional as F

from model.baselearner import BaseLearner1
from data.getdataloader import get_hyperplane2b
from data.getdataloader import get_sea2b

batch_size1 = 2000
batch_size2 = 100
T = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def finetuning1():
    dl = get_hyperplane2b(batch_size1)
    model = BaseLearner1(10, 5, 2)
    model.load_state_dict(torch.load('data/hp2.pt'))
    model.double()
    model.to(device)

    total, corr = 0, 0
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        x = x.view(-1, batch_size2, 10)
        y = y.to(device)
        y = y.long()
        y = y.view(-1, batch_size2)

        for j in range(x.size(0)):
            ybar = model(x[j])
            ybar = ybar.max(1)[1]
            ybar = ybar.view(-1)
            total += batch_size2
            corr += (ybar == y[j]).sum().item()
            for t in range(T):
                model.zero_grad()
                ybar = model(x[j])
                loss = F.nll_loss(ybar, y[j])
                loss.backward()
                for para in model.parameters():
                    para.data.sub_(para.grad.data*0.01)
    print('hyperplane acc:{:.2f}%'.format(corr/total*100))

    dl = get_sea2b(batch_size1)
    model = BaseLearner1(3, 5, 2)
    model.load_state_dict(torch.load('data/sea2.pt'))
    model.double()
    model.to(device)

    total, corr = 0, 0
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        x = x.view(-1, batch_size2, 3)
        y = y.to(device)
        y = y.long()
        y = y.view(-1, batch_size2)

        for j in range(x.size(0)):
            ybar = model(x[j])
            ybar = ybar.max(1)[1]
            ybar = ybar.view(-1)
            total += batch_size2
            corr += (ybar == y[j]).sum().item()
            for t in range(T):
                model.zero_grad()
                ybar = model(x[j])
                loss = F.nll_loss(ybar, y[j])
                loss.backward()
                for para in model.parameters():
                    para.data.sub_(para.grad.data*0.01)
    print('sea acc:{:.2f}%'.format(corr/total*100))
