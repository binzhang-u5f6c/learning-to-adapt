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

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    x = pd.get_dummies(x)
    cols = y.keys()
    ybar = y.pop(cols[0])
    classes = pd.unique(ybar)
    replace_map = {k: i for i, k in enumerate(classes)}
    x[cols[0]] = ybar.replace(replace_map)
    x.to_csv(filename[:-5]+'.csv')


if __name__ == "__main__":
    files = ['data/airlines.arff',
             'data/covtype.arff',
             'data/kddcup99.arff',
             'data/pokerhand.arff']
    for f in files:
        arff2pandas(f)
