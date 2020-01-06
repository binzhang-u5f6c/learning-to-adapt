from math import exp
from math import log

from yaml import safe_load
import torch

with open('config.yml', 'r') as f:
    conf = safe_load(f)
p = conf['p']


def preprocess(x, loss):
    col1 = x.abs().log() / p
    col1[x.abs() < exp(-p)] = -1
    col2 = x.sign()
    col2[x.abs() < exp(-p)] = x[x.abs() < exp(-p)] * exp(p)
    if abs(loss) >= exp(-p):
        col3 = torch.ones(x.size()) * (log(abs(loss))/p)
        col4 = torch.ones(x.size()) * (loss/abs(loss))
    else:
        col3 = torch.ones(x.size()) * (-1)
        col4 = torch.ones(x.size()) * (exp(p)*loss)
    re = torch.cat([col1, col2, col3, col4], 1)
    return re.reshape(-1, 1, 4)
