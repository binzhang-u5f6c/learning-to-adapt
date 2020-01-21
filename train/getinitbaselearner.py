import torch
import torch.nn.functional as F
import torch.optim as optim

from model.baselearner import BaseLearner1
from data.getdataloader import get_hyperplane1a
from data.getdataloader import get_hyperplane2a
from data.getdataloader import get_sea1a
from data.getdataloader import get_sea2a

batch_size = 100
epoch = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dl = get_hyperplane1a(batch_size)
model = BaseLearner1(10, 5, 2)
model.to(device)
model.double()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for e in range(epoch):
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        y = y.to(device)
        y = y.long()
        y = y.view(-1)

        optimizer.zero_grad()
        ybar = model(x)
        loss = F.nll_loss(ybar, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model/hp1.pt')


dl = get_hyperplane2a(batch_size)
model = BaseLearner1(10, 5, 2)
model.to(device)
model.double()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for e in range(epoch):
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        y = y.to(device)
        y = y.long()
        y = y.view(-1)

        optimizer.zero_grad()
        ybar = model(x)
        loss = F.nll_loss(ybar, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model/hp2.pt')


dl = get_sea1a(batch_size)
model = BaseLearner1(3, 5, 2)
model.to(device)
model.double()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for e in range(epoch):
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        y = y.to(device)
        y = y.long()
        y = y.view(-1)

        optimizer.zero_grad()
        ybar = model(x)
        loss = F.nll_loss(ybar, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model/sea1.pt')


dl = get_sea2a(batch_size)
model = BaseLearner1(3, 5, 2)
model.to(device)
model.double()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for e in range(epoch):
    for x, y in dl:
        x = x.to(device)
        x = x.double()
        y = y.to(device)
        y = y.long()
        y = y.view(-1)

        optimizer.zero_grad()
        ybar = model(x)
        loss = F.nll_loss(ybar, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model/sea2.pt')
