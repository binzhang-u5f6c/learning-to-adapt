from train.getinitbaselearner import get_init_baselearner

files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']


for filename in files:
    get_init_baselearner(filename)
