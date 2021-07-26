import torch
from torch import nn
import torch.nn.functional as F


class MyBatchNorm(nn.Module):
    def __init__(self, n_features, n_dims, momentum=0.999):
        super(MyBatchNorm, self).__init__()
        if n_dims == 1:
            _shape = (1, n_features)
        elif n_dims == 2:
            _shape = (1, n_features, 1, 1)
        else:
            raise ValueError("MyBatchNorm only works for linear layers and 2D convs")

        # hyper-parameters, different tasks might benefit from custom values for momentum, whereas epsilon can be hard-coded
        self.momentum = momentum
        self.eps = 1e-7

        # learnable parameters, gamma is initialized at 1 (will be used for scaling) and beta is initialized at 0 (will be used for shifting)
        self.gamma = nn.Parameter(torch.ones(size=_shape))
        self.beta = nn.Parameter(torch.zeros(size=_shape))
        # dataset statistics
        self.stats_E = None
        self.stas_var = None

    def forward(self, x):
        if self.training:
            return self._training_forward(x)
        else:
            return self._inference_forward(x)

    def _training_forward(self, x):
        # compute batch statistics
        E = torch.mean(x)
        var = torch.var(x)

        # update dataset statistics
        self.update_stats(E, var)

        # normalize data
        x = (x - E) / torch.sqrt(var + self.eps)

        # return scaled and shifted tensors
        return x * self.gamma + self.beta

    def _inference_forward(self, x):
        # normalize data by means of dataset statistics (computed during training)
        x = (x - self.stats_E) / torch.sqrt(self.stats_var + self.eps)

        # return scaled and shifted tensors
        return x * self.gamma + self.beta

    def update_stats(self, batch_E, batch_var):
        with torch.no_grad():
            # statistics are initialized to the first input data
            if self.stats_E is None:
                self.stats_E = batch_E
                self.stats_var = batch_var

            # statistics are updated by means of Exponential Moving Average
            self.stats_E = self.stats_E * (1 - self.momentum) + batch_E * self.momentum
            self.stats_var = self.stats_var * (1 - self.momentum) + batch_var * self.momentum


# LeNet implementation from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
class LeNet(nn.Module):

    def __init__(self, bn=False):
        super(LeNet, self).__init__()

        if bn:
            self.bn = MyBatchNorm
        # Using nn.Identity allows for cleaner code
        else:
            self.bn = nn.Identity

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn_conv1 = self.bn(n_features=6, n_dims=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn_conv2 = self.bn(n_features=16, n_dims=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn_fc1 = self.bn(n_features=120, n_dims=1)
        self.fc2 = nn.Linear(120, 84)
        self.bn_fc2 = self.bn(n_features=84, n_dims=1)
        self.fc3 = nn.Linear(84, 10)

    # Batch Norm is added after each layer but the last one, before the activation function
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn_conv1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn_conv2((self.conv2(x)))), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x
