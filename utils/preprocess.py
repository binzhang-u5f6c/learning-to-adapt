from math import exp
from math import log

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(x, loss, p):
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
