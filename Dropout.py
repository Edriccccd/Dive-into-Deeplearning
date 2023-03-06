import torch
from torch import nn
from d2l import torch as d2l

def dropout_layer(X, dropout):
    assert 0<= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)



num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256