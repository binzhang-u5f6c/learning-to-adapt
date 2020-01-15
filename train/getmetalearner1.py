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
         'data/pokerhand.arff']


def get_metalearner1(filename, batch_size1, batch_size2, hidden_size,
                     m_hidden_size, training_size, epoch, lr, T, p):
    metalearner = MetaLearner(m_hidden_size)
    metalearner.to(device)
    metalearner.double()
    optimizer = optim.SGD(metalearner.parameters(), lr=lr, momentum=0.9)

    for e in range(epoch):
        for f in files:
            if f == filename:
                continue
            dataloader, input_size, output_size = \
                get_dataloader(f, batch_size1, training_size, False)

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
                batch_x = batch_x.view(-1, batch_size2, input_size)
                batch_y = batch_y.to(device)
                batch_y = batch_y.long()
                batch_y = batch_y.view(-1, batch_size2)

                optimizer.zero_grad()
                for j in range(batch_x.size(0)-1):
                    for t in range(T):
                        model_cp.load_state_dict(model.state_dict())
                        grad, loss = get_grad(model_cp, batch_x[j], batch_y[j])
                        hc = [None for i in grad]
                        for n, para in enumerate(model.parameters()):
                            meta_input = preprocess(grad[n], loss, p)
                            meta_input.to(device)
                            if hc[n] is None:
                                hc[n] = (torch.randn(2, meta_input.size(0),
                                                     m_hidden_size,
                                                     device=device,
                                                     dtype=torch.float64),
                                         torch.randn(2, meta_input.size(0),
                                                     m_hidden_size,
                                                     device=device,
                                                     dtype=torch.float64))
                            meta_output, hc[n] = \
                                metalearner(meta_input, hc[n])

                            meta_output = meta_output.view(para.data.size())
                            para.data.sub_(meta_output)

                    ybar = model(batch_x[j+1])
                    t_loss = F.nll_loss(ybar, batch_y[j+1])
                    t_loss.backward()
                    optimizer.step()

    torch.save(metalearner.state_dict(), 'meta1.'+filename[5:]+'.pt')
