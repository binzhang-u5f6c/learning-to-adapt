import torch.nn.functional as F


def get_grad(model, x, y):
    model.zero_grad()
    ybar = model(x)
    loss = F.nll_loss(ybar, y)
    loss.backward()
    grad = [i.grad.data.view(-1, 1) for i in model.parameters()]
    return grad, loss.item()
