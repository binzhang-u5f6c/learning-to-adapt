import torch
import torch.nn.functional as F
import torch.optim as optim

from model.baselearner import BaseLearner1
from model.metalearner import MetaLearner
from data.getdataloader import get_hyperplane1b
from data.getdataloader import get_sea1b
from utils.getgrad import get_grad
from utils.preprocess import preprocess

batch_size1 = 2000
batch_size2 = 100
epoch = 10
T = 10
p = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

metalearner = MetaLearner(5)
metalearner.to(device)
metalearner.double()
optimizer = optim.SGD(metalearner.parameters(), lr=0.01, momentum=0.9)

dl1 = get_hyperplane1b(batch_size1)
dl2 = get_sea1b(batch_size1)

for e in range(epoch):
    model = BaseLearner1(10, 5, 2)
    model.load_state_dict(torch.load('data/hp1.pt'))
    model.double()
    model.to(device)

    model_cp = BaseLearner1(10, 5, 2)
    model_cp.double()
    model_cp.to(device)

    for x, y in dl1:
        x = x.to(device)
        x = x.double()
        x = x.view(-1, batch_size2, 10)
        y = y.to(device)
        y = y.long()
        y = y.view(-1, batch_size2)

        for j in range(x.size(0)):
            optimizer.zero_grad()
            for t in range(T):
                model_cp.load_state_dict(model.state_dict())
                grad, loss = get_grad(model_cp, x[j][:50], y[j][:50])
                hc = [None for i in grad]
                for n, para in enumerate(model.parameters()):
                    meta_input = preprocess(grad[n], loss, p)
                    meta_input.to(device)
                    if hc[n] is None:
                        hc[n] = (torch.zeros(2, meta_input.size(0),
                                             5, device=device,
                                             dtype=torch.float64),
                                 torch.zeros(2, meta_input.size(0),
                                             5, device=device,
                                             dtype=torch.float64))
                    meta_output, hc[n] = metalearner(meta_input, hc[n])

                    meta_output = meta_output.view(para.data.size())
                    para.data.sub_(meta_output)
            l2 = 0
            for para in model.parameters():
                l2 += (para**2).sum()
            ybar = model(x[j][50:])
            t_loss = F.nll_loss(ybar, y[j][50:]) + l2
            t_loss.backward()
            torch.nn.utils.clip_grad_value_(metalearner.parameters(), 10)
            optimizer.step()

    model = BaseLearner1(3, 5, 2)
    model.load_state_dict(torch.load('data/sea1.pt'))
    model.double()
    model.to(device)

    model_cp = BaseLearner1(3, 5, 2)
    model_cp.double()
    model_cp.to(device)

    for x, y in dl2:
        x = x.to(device)
        x = x.double()
        x = x.view(-1, batch_size2, 10)
        y = y.to(device)
        y = y.long()
        y = y.view(-1, batch_size2)

        for j in range(x.size(0)):
            optimizer.zero_grad()
            for t in range(T):
                model_cp.load_state_dict(model.state_dict())
                grad, loss = get_grad(model_cp, x[j][:50], y[j][:50])
                hc = [None for i in grad]
                for n, para in enumerate(model.parameters()):
                    meta_input = preprocess(grad[n], loss, p)
                    meta_input.to(device)
                    if hc[n] is None:
                        hc[n] = (torch.zeros(2, meta_input.size(0),
                                             5, device=device,
                                             dtype=torch.float64),
                                 torch.zeros(2, meta_input.size(0),
                                             5, device=device,
                                             dtype=torch.float64))
                    meta_output, hc[n] = metalearner(meta_input, hc[n])

                    meta_output = meta_output.view(para.data.size())
                    para.data.sub_(meta_output)
            l2 = 0
            for para in model.parameters():
                l2 += (para**2).sum()
            ybar = model(x[j][50:])
            t_loss = F.nll_loss(ybar, y[j][50:]) + l2
            t_loss.backward()
            torch.nn.utils.clip_grad_value_(metalearner.parameters(), 10)
            optimizer.step()

torch.save(metalearner.state_dict(), 'meta.pt')
