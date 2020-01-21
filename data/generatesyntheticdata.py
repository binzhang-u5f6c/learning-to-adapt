from random import randint
from random import uniform

import numpy as np
import pandas as pd
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import SEAGenerator

# generate data stream for training meta model
generator_hp = HyperplaneGenerator(random_state=0)
generator_hp.prepare_for_use()
x, y = generator_hp.next_sample(10000)
y = y.reshape((-1, 1))
df = pd.DataFrame(np.hstack((x, y)))
df.to_csv('data/hyperplane.1a.csv')

d = np.ones((1, 11))
drift_flag = False
for i in range(10000):
    if not drift_flag:
        rd = uniform(0, 1)
        generator_hp.set_params(mag_change=rd)
        x, y = generator_hp.next_sample()
        generator_hp.set_params(mag_change=0)
        drift_flag = True
    else:
        drift_flag = False
    x, y = generator_hp.next_sample(100)
    y = y.reshape((-1, 1))
    d = np.vstack((d, np.hstack((x, y))))
df = pd.DataFrame(d)
df.to_csv('data/hyperplane.1b.csv')

generator_sea = SEAGenerator(random_state=0)
generator_sea.prepare_for_use()
x, y = generator_sea.next_sample(10000)
y = y.reshape((-1, 1))
df = pd.DataFrame(np.hstack((x, y)))
df.to_csv('data/sea.1a.csv')

d = np.ones((1, 4))
drift_flag = False
for i in range(10000):
    if not drift_flag:
        rd = randint(0, 3)
        generator_sea.set_params(classification_function=rd)
        drift_flag = True
    else:
        drift_flag = False
    x, y = generator_sea.next_sample(100)
    y = y.reshape((-1, 1))
    d = np.vstack((d, np.hstack((x, y))))
df = pd.DataFrame(d)
df.to_csv('data/sea.1b.csv')

# generate data stream for evaluating meta model
generator_hp = HyperplaneGenerator(random_state=65535)
generator_hp.prepare_for_use()
x, y = generator_hp.next_sample(10000)
y = y.reshape((-1, 1))
df = pd.DataFrame(np.hstack((x, y)))
df.to_csv('data/hyperplane.2a.csv')

d = np.ones((1, 11))
drift_flag = False
for i in range(10000):
    if not drift_flag:
        rd = uniform(0, 1)
        generator_hp.set_params(mag_change=rd)
        x, y = generator_hp.next_sample()
        generator_hp.set_params(mag_change=0)
        drift_flag = True
    else:
        drift_flag = False
    x, y = generator_hp.next_sample(100)
    y = y.reshape((-1, 1))
    d = np.vstack((d, np.hstack((x, y))))
df = pd.DataFrame(d)
df.to_csv('data/hyperplane.2b.csv')

generator_sea = SEAGenerator(random_state=65535)
generator_sea.prepare_for_use()
x, y = generator_sea.next_sample(10000)
y = y.reshape((-1, 1))
df = pd.DataFrame(np.hstack((x, y)))
df.to_csv('data/sea.2a.csv')

d = np.ones((1, 4))
drift_flag = False
for i in range(10000):
    if not drift_flag:
        rd = randint(0, 3)
        generator_sea.set_params(classification_function=rd)
        drift_flag = True
    else:
        drift_flag = False
    x, y = generator_sea.next_sample(100)
    y = y.reshape((-1, 1))
    d = np.vstack((d, np.hstack((x, y))))
df = pd.DataFrame(d)
df.to_csv('data/sea.2b.csv')
