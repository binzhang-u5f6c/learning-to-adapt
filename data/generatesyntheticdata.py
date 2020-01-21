import numpy as np
import pandas as pd
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import SEAGenerator

generator_hp = HyperplaneGenerator(random_state=0)
generator_hp.prepare_for_use()
x, y = generator_hp.next_sample(10000)
y = y.reshape((-1, 1))
df = pd.DataFrame(np.hstack((x, y)))
df.to_csv('data/hyperplane.1a.csv')

d = np.ones((1, 11))
for i in range(10000):
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
for i in range(10000):
    x, y = generator_sea.next_sample(100)
    y = y.reshape((-1, 1))
    d = np.vstack((d, np.hstack((x, y))))
df = pd.DataFrame(d)
df.to_csv('data/sea.1b.csv')
