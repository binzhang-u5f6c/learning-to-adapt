from train.getinitbaselearner import get_init_baselearner
from train.getmetalearner import get_metalearner

files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']


for filename in files:
    # get_init_baselearner(filename)
    get_metalearner(filename)
