from time import asctime

from train.getinitbaselearner import get_init_baselearner
from train.getmetalearner import get_metalearner
from evaluate.baseline import baseline
from evaluate.metaadapt import meta_adapt

files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']


print(asctime())
for filename in files:
    get_init_baselearner(filename)
for filename in files:
    get_metalearner(filename)
print(asctime())
for filename in files:
    print(filename)
    corr, total = baseline(filename)
    print(corr/total)
    corr, total = meta_adapt
    print(corr/total)
