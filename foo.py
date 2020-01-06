import torch

from model.baselearner import BaseLearner
from utils.getgrad import get_grad

model = BaseLearner(3, 3, 2)
x = torch.ones(3, 3)
y = torch.ones(3)
y = y.long()

i = get_grad(model, x, y)
print(i)
