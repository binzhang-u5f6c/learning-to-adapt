from yaml import safe_load
import torch

from model.baselearner import BaseLearner
from model.metalearner import MetaLearner
from utils.load import get_dataloader
from utils.getgrad import get_grad
from utils.preprocess import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config.yml', 'r') as f:
    conf = safe_load(f)
meta_hidden_size = conf['meta_hidden_size']
T = conf['T']


def meta_adapt(filename):
    dataloader, input_size, output_size = get_dataloader(filename, False)

    hidden_size = int((input_size + output_size) / 2)
    model = BaseLearner(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename[5:]+'.pt'))
    model.to(device)

    model_cp = BaseLearner(input_size, hidden_size, output_size)
    model_cp.double()
    model_cp.to(device)

    metalearner = MetaLearner(meta_hidden_size)
    metalearner.load_state_dict(torch.load(filename[5:]+'.meta.pt'))
    metalearner.to(device)

    total = 0
    correct = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_x = batch_x.double()
        batch_y = batch_y.to(device)
        batch_y = batch_y.long()
        batch_y = batch_y.view(-1)

        for t in range(T):
            model_cp.load_state_dict(model.state_dict())
            grad, loss = get_grad(model_cp, batch_x, batch_y)
            hc1 = [None for i in grad]
            hc2 = [None for i in grad]
            for n, para in enumerate(model.parameters()):
                meta_input = preprocess(grad[n], loss)
                meta_input.to(device)
                if hc1[n] is None:
                    hc1[n] = (torch.randn(1, meta_input.size(0),
                                          meta_hidden_size,
                                          device=device),
                              torch.randn(1, meta_input.size(0),
                                          meta_hidden_size,
                                          device=device))
                if hc2[n] is None:
                    hc2[n] = (torch.randn(1, meta_input.size(0), 2,
                                          device=device),
                              torch.randn(1, meta_input.size(0), 2,
                                          device=device))
                meta_output, hc1[n], hc2[n] = \
                    metalearner(meta_input, hc1[n], hc2[n])

                ft = meta_output[:, :, 0].reshape(para.data.size())
                it = meta_output[:, :, 1].reshape(para.data.size())
                para.data.mul_(ft)
                para.data.add_(it)

                ybar = model(batch_x)
                ybar = ybar.max(1)[1]
                ybar = ybar.view(-1)
                total += batch_y.size(0)
                correct += (ybar == batch_y).sum().item()
    return correct, total
