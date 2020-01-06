from time import asctime

from train.getinitbaselearner import get_init_baselearner
from train.getmetalearner import get_metalearner

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
