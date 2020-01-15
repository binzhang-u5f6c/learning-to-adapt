import torch

from model.baselearner import BaseLearner
from model.metalearner import MetaLearner
from utils.load import get_dataloader
from utils.getgrad import get_grad
from utils.preprocess import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def lftee1(filename, batch_size1, batch_size2, hidden_size,
           meta_hidden_size, training_size, T, p):
    dataloader, input_size, output_size = get_dataloader(filename, batch_size1,
                                                         training_size, False)

    model = BaseLearner(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename[5:]+'.pt'))
    model.double()
    model.to(device)

    model_cp = BaseLearner(input_size, hidden_size, output_size)
    model_cp.double()
    model_cp.to(device)

    metalearner = MetaLearner(meta_hidden_size)
    metalearner.load_state_dict(torch.load('meta1.'+filename[5:]+'.pt'))
    metalearner.double()
    metalearner.to(device)

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
                model_cp.load_state_dict(model.state_dict())
                grad, loss = get_grad(model_cp, batch_x[j], batch_y[j])
                hc = [None for i in grad]
                for n, para in enumerate(model.parameters()):
                    meta_input = preprocess(grad[n], loss, p)
                    meta_input.to(device)
                    if hc[n] is None:
                        hc[n] = (torch.randn(2, meta_input.size(0),
                                             meta_hidden_size,
                                             device=device,
                                             dtype=torch.float64),
                                 torch.randn(2, meta_input.size(0),
                                             meta_hidden_size,
                                             device=device,
                                             dtype=torch.float64))
                    meta_output, hc[n] = \
                        metalearner(meta_input, hc[n])

                    meta_output = meta_output.view(para.data.size())
                    para.data.sub_(meta_output)

    return correct, total
