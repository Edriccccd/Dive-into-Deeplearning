import torch
from torch import nn
from d2l import torch as d2l

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.001)



batch_size = 512
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(), nn.Linear(784,10))
net.apply(init_weights);
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr = 0.05)
num_epochs = 20
d2l.train_ch3(net, train_iter, test_iter, loss,num_epochs, trainer)
d2l.plt.show()