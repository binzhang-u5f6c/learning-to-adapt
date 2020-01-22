import torch

from model.baselearner import BaseLearner1
from model.metalearner import MetaLearner
from data.getdataloader import get_hyperplane2b
from data.getdataloader import get_sea2b
from utils.getgrad import get_grad
from utils.preprocess import preprocess

batch_size1 = 2000
batch_size2 = 100
T = 10
p = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lftee1():
    metalearner = MetaLearner(5)
    metalearner.load_state_dict(torch.load('model/meta.pt'))
    metalearner.to(device)
    metalearner.double()

    dl = get_hyperplane2b(batch_size1)
    model = BaseLearner1(10, 5, 2)
    model.load_state_dict(torch.load('data/hp2.pt'))
    model.double()
    model.to(device)

    model_cp = BaseLearner1(10, 5, 2)
    model_cp.double()
    model_cp.to(device)

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
                model_cp.load_state_dict(model.state_dict())
                grad, loss = get_grad(model_cp, x[j], y[j])
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
    print('hyperplane acc:{:.2f}%'.format(corr/total*100))

    dl = get_sea2b(batch_size1)
    model = BaseLearner1(3, 5, 2)
    model.load_state_dict(torch.load('data/sea2.pt'))
    model.double()
    model.to(device)

    model_cp = BaseLearner1(3, 5, 2)
    model_cp.double()
    model_cp.to(device)

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
                model_cp.load_state_dict(model.state_dict())
                grad, loss = get_grad(model_cp, x[j], y[j])
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
    print('sea acc:{:.2f}%'.format(corr/total*100))
