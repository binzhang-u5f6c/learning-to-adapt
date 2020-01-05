import numpy as np
import pandas as pd
from scipy.io import arff


def arff2pandas(filename):
    data_raw, meta = arff.loadarff(filename)
    seri_dict = {}
    n = len(meta.names())
    for i in range(n):
        name = meta.names()[i]
        d_type = meta.types()[i]
        d_list = [j[i] for j in data_raw]
        if d_type == 'numeric':
            seri_dict[name] = pd.Series(d_list, dtype=np.float)
        else:
            seri_dict[name] = pd.Series(d_list, dtype='category')
    df = pd.DataFrame(seri_dict)
    return df
