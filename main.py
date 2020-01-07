from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.load import DatasetFoo

files = ['data/airlines.arff',
         'data/covtype.arff',
         'data/kddcup99.arff',
         'data/pokerhand.arff',
         'data/sensor.arff']

for filename in files:
    ds1 = DatasetFoo(filename)
    x = ds1.x
    y = ds1.y
    clf = LogisticRegression()
    clf.fit(x, y)
    ds1 = DatasetFoo(filename, False)
    x = ds1.x
    y = ds1.y
    ybar = clf.predict(x)
    acc = accuracy_score(ybar, y)
