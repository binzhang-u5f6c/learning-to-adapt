from yaml import safe_load
import torch
import torch.nn.functional as F
import torch.optim as optim

from model.baselearner import BaseLearner
from model.metalearner import MetaLearner
from utils.load import get_dataloader
from utils.getgrad import get_grad
from utils.preprocess import preprocess

# hyperparemeters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']
with open('config.yml', 'r') as f:
    conf = safe_load(f)
epoch = conf['m_epoch']
lr = conf['m_lr']
m_hidden_size = conf['meta_hidden_size']
T = conf['T']


def get_metalearner(filename):
    metalearner = MetaLearner(m_hidden_size)
    metalearner.to(device)
    metalearner.double()
    optimizer = optim.SGD(metalearner.parameters(), lr=lr, momentum=0.9)

    for e in range(epoch):
        for f in files:
            if f == filename:
                continue
            dataloader, input_size, output_size = \
                get_dataloader(f, False)

            hidden_size = int((input_size + output_size) / 2)
            model = BaseLearner(input_size, hidden_size, output_size)
            model.load_state_dict(torch.load(f[5:]+'.pt'))
            model.double()
            model.to(device)

            model_cp = BaseLearner(input_size, hidden_size, output_size)
            model_cp.double()
            model_cp.to(device)

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_x = batch_x.double()
                batch_y = batch_y.to(device)
                batch_y = batch_y.long()
                batch_y = batch_y.view(-1)

                optimizer.zero_grad()
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
                                                  m_hidden_size,
                                                  device=device,
                                                  dtype=torch.float64),
                                      torch.randn(1, meta_input.size(0),
                                                  m_hidden_size,
                                                  device=device,
                                                  dtype=torch.float64))
                        if hc2[n] is None:
                            hc2[n] = (torch.randn(1, meta_input.size(0), 2,
                                                  device=device,
                                                  dtype=torch.float64),
                                      torch.randn(1, meta_input.size(0), 2,
                                                  device=device,
                                                  dtype=torch.float64))
                        meta_output, hc1[n], hc2[n] = \
                            metalearner(meta_input, hc1[n], hc2[n])

                        ft = meta_output[:, :, 0].reshape(para.data.size())
                        it = meta_output[:, :, 1].reshape(para.data.size())
                        para.data.mul_(ft)
                        para.data.add_(it)

                ybar = model(batch_x)
                t_loss = F.nll_loss(ybar, batch_y)
                t_loss.backward()
                optimizer.step()
            with open('output.txt', 'w') as f:
                f.write('batch test completed!')
            exit()

    torch.save(metalearner.state_dict(), filename[5:]+'.meta.pt')
