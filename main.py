from train.getinitbaselearner import get_init_baselearner
from train.getmetalearner import get_metalearner
from evaluate.baseline import baseline
from evaluate.lftee import lftee

# hyperparameters
files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff']
training_size = 100000
batch_size1 = 10000
batch_size2 = 100

b_hidden_size = 5
m_hidden_size = 5

# b_epoch = 200
b_lr = 0.01

m_epoch = 3
m_lr = 0.01
T = 10
p = 10

# train initial model
get_init_baselearner('data/airlines.arff', b_hidden_size, batch_size1,
                     training_size, 200, b_lr)
get_init_baselearner('data/covtype.arff', b_hidden_size, batch_size1,
                     training_size, 200, b_lr)
get_init_baselearner('data/kddcup99.arff', b_hidden_size, batch_size1,
                     training_size, 500, b_lr)
get_init_baselearner('data/pokerhand.arff', b_hidden_size, batch_size1,
                     training_size, 200, b_lr)

# baseline
for filename in files:
    corr, total = baseline(filename, b_hidden_size,
                           batch_size1, training_size)
    print(' '+filename+': ', corr/total)

# lftee
for filename in files:
    get_metalearner(filename, batch_size1, batch_size2, b_hidden_size,
                    m_hidden_size, training_size, m_epoch, m_lr, T, p)
    corr, total = lftee(filename, batch_size1, batch_size2, b_hidden_size,
                        m_hidden_size, training_size, T, p)
    print(' '+filename+': ', corr/total)
