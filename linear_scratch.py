import random
import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l

def synthetic_data(w, b, num_example):
    X = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(X,w)+b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))                    #reshape用来重新生成矩阵，（-1,1）其中-1代表行数是随便的，直接规划成n行1列矩阵


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linrge(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) **  2/2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
            
            

            
true_w = torch.tensor([200, -33.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10


w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.005
num_epochs = 30
net = linrge
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b),y)
        l.sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        # print(f'epoch {epoch + 1}, loss {float(train_l.mean():f)}')
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(f'w的误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的误差: {true_b - b}')


# print('features:', features[0], '\nlabel', labels[0])

# plt.figure(figsize=(5,5))
# plt.scatter(features[:, 1].detach().numpy(), labels.detach(), 10, c="#88c999", edgecolors="green")
# plt.show()