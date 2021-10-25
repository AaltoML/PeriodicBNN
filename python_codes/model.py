"""
    MLP model definition
"""

import math
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F
from scipy.special import gamma
import numpy as np
from googlenet import Inception, BasicConv2d
import os

torch.pi = 3.1415926535897932
torch.pi2 = 6.2831853071795864
torch.sqrt2 = 1.414213562    # sqrt 2 \approx 1.414213562
torch.pdiv2sqrt2 = 1.1107207345     # π/(2*sqrt(2)) \approx 1.1107207345
torch.pdiv2 = 1.570796326 # π/2
torch.pdiv4 = 0.785398163 # π/4 

def sincos_activation(x):
    return torch.sin(x) + torch.cos(x)

def sin_activation(x):
    return torch.sqrt2*torch.sin(x)

def _triangle_activation(x):
    return (x - torch.pi * torch.floor(x / torch.pi + 0.5)) * (-1)**torch.floor(x/torch.pi + 0.5)

def triangle_activation(x):
    return torch.pdiv2sqrt2 * _triangle_activation(x)

def periodic_relu_activation(x):
    return torch.pdiv4 * (_triangle_activation(x) + _triangle_activation(x + torch.pdiv2))

def rbf_activation(x):
    return torch.exp(-1*(x)**2)

def invlink_uniform(x):
    if x is not None:
        return torch.pi2*torch.sigmoid(x) - torch.pi
    else:
        return x

class single_param(nn.Module):

    def __init__(self, value):
        super(single_param, self).__init__()
        self.p = nn.Parameter(torch.FloatTensor([value]))

    def forward(self):
        return torch.abs(self.p)

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) aka neural network for classification tasks.
    MLP(torch.nn.Module)

    ...

    Attributes
    ----------
    num_classes : int
        number of classes
    K : int
        number of hidden units in the last hidden layer (default = 50)
    pipeline : nn.Module
        the feature extractor part of the NN architecture preceding the model layer
    lengthscale : int
        length scale parameter (default = 1)
    dropout : float
        dropout rate (default = 0.0)
    kernel : str
        kernel function: Matern, RBF or ArcCosine (ReLU activation) (default = Matern)
    periodic_fun : str
        periodic activation to use: sin, triangle, sincos, or prelu (default = sin) This is only used if global_stationary=True!
    global_stationary : bool
        Use global stationarity inducing activation function (default = True)
    nu : float
        Matern parameter (default = 3/2)
    device : str
        device (default = cpu)
    """
    def __init__(self, num_classes=2, D = 1, K=50, pipeline=None, lengthscale = 1, dropout=0.0, 
        kernel = 'Matern', periodic_fun = 'sin', global_stationary = True,
        nu = 3/2, device = 'cpu', meas_noise = False):

        super(MLP, self).__init__()

        #FC layers
        self.pipeline = pipeline(D = D, dropout=dropout)
        self.K = K

        if pipeline == CIFAR_PIPELINE:
            print("Loading pretrained model")
            
            # Pretrained model available at https://github.com/huyvnphan/PyTorch_CIFAR10
            state_dict = torch.load('../state_dicts/updated_googlenet.pt', map_location=device)
            self.pipeline.load_state_dict(state_dict, strict = False)
            
            for param in self.pipeline.parameters():
                param.requires_grad = False

        self.fc_o = nn.Linear(K, num_classes)
        self.drop_layer = nn.Dropout(p=dropout)
        
        self.lengthscale = single_param(lengthscale)
        self.l_dist = torch.distributions.gamma.Gamma(torch.tensor(2.0).to(device), torch.tensor(1/2).to(device))

        self.nu = nu
        
        if meas_noise:
            self.s = single_param(meas_noise)
            self.s_dist = torch.distributions.gamma.Gamma(torch.tensor(0.5).to(device), torch.tensor(1.0).to(device))
        else:
            self.s = None

        if global_stationary:

            bias = True
            if periodic_fun == 'triangle':
                self.activation = triangle_activation
            elif periodic_fun == 'prelu':
                self.activation = periodic_relu_activation
            elif periodic_fun == 'sin':
                self.activation = sin_activation
            elif periodic_fun == 'sincos':
                bias = False
                self.activation = sincos_activation
            else:
                raise Exception("Unknown periodic function! Available functions: [sin, triangle].")

            self.fc_h = ConstrainedLinear(self.pipeline.O, K, bias)

            if kernel == 'Matern':
                self.Pw_dist = torch.distributions.studentT.StudentT(2*nu)
            elif kernel == 'RBF':
                self.Pw_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
            else:
                 raise Exception("Unknown kernel function! Available functions: [Matern, RBF].")
            
            pi = torch.tensor(torch.pi).to(device)

            if bias:
                self.Pb_dist = torch.distributions.uniform.Uniform(-pi, pi)
            else:
                self.register_parameter('Pb_dist', None)

            print("# Constructing globally stationary MLP with num_classes={}, K={}, kernel={}, periodic fun={}\
            ".format(num_classes, K, kernel, periodic_fun))

        else:

            if kernel == 'Matern':
                self.activation = LocaLMatern(nu, device)
            elif kernel == 'RBF':
                self.activation = rbf_activation
            elif kernel == 'ArcCosine':
                self.activation = nn.ReLU()
            else:
                 raise Exception("Unknown kernel function! Available functions: [Matern, RBF, ArcCosine].")

            self.fc_h = nn.Linear(self.pipeline.O, K)

            self.Pw_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
            self.Pb_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

            print("# Constructing MLP with num_classes={}, D={}, K={}, kernel/activation={}\
            ".format(num_classes, D, K, kernel, periodic_fun))

        self.Pw_o_dist = torch.distributions.normal.Normal(loc=0.0, scale=1.0/(K**0.5))

        self.init_weights()
        
    def init_weights(self):
        self.fc_h.weight = nn.Parameter(self.Pw_dist.sample(self.fc_h.weight.shape))
        self.fc_o.weight = nn.Parameter(self.Pw_o_dist.sample(self.fc_o.weight.shape))
        if self.fc_h.bias is not None:
            self.fc_h.bias = nn.Parameter(self.Pb_dist.sample(self.fc_h.bias.shape))
        
    def dropout_off(self):
        self.drop_layer.p = 0

    def forward(self, x):

        x = self.pipeline(x)
        x = x * self.lengthscale.forward()
        x = self.fc_h(x)
        x = self.activation(x)
        x = self.fc_o(x)
        return x


class LocaLMatern(nn.Module):

    def __init__(self, nu, device):
        super(LocaLMatern, self).__init__()
        self.nu = nu
        self.A = torch.sqrt(2*torch.pi**0.5*(2*nu)**nu/torch.from_numpy(np.array([gamma(nu)*gamma(nu+0.5)]))).to(device)

    def forward(self, x):
        y = self.A/(self.nu)*torch.sign(x)*torch.abs(x)**(self.nu-0.5)*torch.exp(-(2*self.nu)**0.5*torch.abs(x))
        y[x<0] = 0
        return y.float()

class ConstrainedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias :bool):
        super(ConstrainedLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = F.linear(x, self.weight, invlink_uniform(self.bias))
        return x
        

class BANANA_PIPELINE(nn.Module):
    def __init__(self, D = 5, dropout = 0.0):
        super(BANANA_PIPELINE, self).__init__()
        
        self.O = D

    def forward(self, x):
        return x

class UCI_PIPELINE(nn.Module):

    def __init__(self, D = 5, dropout = 0.0):
        super(UCI_PIPELINE, self).__init__()

        self.O = 25

        self.fc1 = nn.Linear(D, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, self.O)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        return x

class MNIST_PIPELINE(nn.Module):

    def __init__(self, D = 5, dropout = 0.25):
        super(MNIST_PIPELINE, self).__init__()

        self.O = 25
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(9216, self.O)        

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        
        #Additional bottleneck
        x = self.linear(x)
        x = F.relu(x)
        
        return x

class CIFAR_PIPELINE(nn.Module):

    def __init__(self, D = 5, dropout = 0.0, pretrained = True):
        super(CIFAR_PIPELINE, self).__init__()
        self.conv1 = BasicConv2d(3, 192, kernel_size=3, stride=1, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)
        
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.O = 25
        self.linear = nn.Linear(1024, self.O)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        # N x 3 x 224 x 224
        x = self.conv1(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        
        #Additional bottleneck
        x = self.linear(x)
        x = F.relu(x)
        
        return x


# Helpers for building models

class cifar:
    base = MLP
    args = list()
    kwargs = dict()
    kwargs['K'] = 2000
    kwargs['pipeline'] = CIFAR_PIPELINE

class banana:
    base = MLP
    args = list()
    kwargs = dict()
    kwargs['K'] = 1000
    kwargs['pipeline'] = BANANA_PIPELINE
    
class uci:
    base = MLP
    args = list()
    kwargs = dict()
    kwargs['K'] = 2000
    kwargs['pipeline'] = UCI_PIPELINE

class mnist:
    base = MLP
    args = list()
    kwargs = dict()
    kwargs['K'] = 2000
    kwargs['pipeline'] = MNIST_PIPELINE
    
class regression_1D:
    base = MLP
    args = list()
    kwargs = dict()
    kwargs['meas_noise'] = 1
    kwargs['pipeline'] = BANANA_PIPELINE
