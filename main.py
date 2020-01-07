from train.getinitbaselearner import get_init_baselearner
from evaluate.baseline import baseline

# hyperparameters
files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']
training_size = 100000
batch_size1 = 1000

b_lr = 0.01

for b_epoch in [200, 500, 1000, 2000, 5000, 10000]:
    print('epoch = {}'.format(b_epoch))
    for filename in files:
        get_init_baselearner(filename, batch_size1,
                             training_size, b_epoch, b_lr)
        corr, total = baseline(filename, batch_size1, training_size)
        print(' '+filename+': ', corr/total)
