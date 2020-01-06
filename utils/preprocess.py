from math import exp
from math import log

from yaml import safe_load
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config.yml', 'r') as f:
    conf = safe_load(f)
p = conf['p']


def preprocess(x, loss):
    col1 = x.abs().log() / p
    col1[x.abs() < exp(-p)] = -1
    col2 = x.sign()
    col2[x.abs() < exp(-p)] = x[x.abs() < exp(-p)] * exp(p)
    if abs(loss) >= exp(-p):
        col3 = (log(abs(loss))/p) * \
            torch.ones(x.size(), device=device, dtype=torch.float64)
        col4 = (loss/abs(loss)) * \
            torch.ones(x.size(), device=device, dtype=torch.float64)
    else:
        col3 = (-1) * \
            torch.ones(x.size(), device=device, dtype=torch.float64)
        col4 = (exp(p)*loss) * \
            torch.ones(x.size(), device=device, dtype=torch.float64)
    re = torch.cat([col1, col2, col3, col4], 1)
    return re.reshape(-1, 1, 4)
