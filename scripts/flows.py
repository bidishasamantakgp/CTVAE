import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import unconstrained_RQS

functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, is_cuda):
        super(FCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )
        if is_cuda:
            self.cuda()
    def forward(self, x):
        #print("x:", x.is_cuda)
        return self.network(x)

class RealNVP(nn.Module):
     """
     Non-volume preserving flow.
     [Dinh et. al. 2017]
     """
     def __init__(self, dim, hidden_dim = 8, base_network=FCNN, is_cuda = True):
         super().__init__()
         self.dim = dim
         self.t1 = base_network(dim // 2, dim // 2, hidden_dim, is_cuda = is_cuda)
         self.s1 = base_network(dim // 2, dim // 2, hidden_dim, is_cuda = is_cuda)
         self.t2 = base_network(dim // 2, dim // 2, hidden_dim, is_cuda = is_cuda)
         self.s2 = base_network(dim // 2, dim // 2, hidden_dim, is_cuda = is_cuda)
         if is_cuda:
              self.cuda()
 
     def forward(self, x):
         lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
         t1_transformed = self.t1(lower)
         s1_transformed = self.s1(lower)
         upper = t1_transformed + upper * torch.exp(s1_transformed)
         t2_transformed = self.t2(upper)
         s2_transformed = self.s2(upper)
         lower = t2_transformed + lower * torch.exp(s2_transformed)
         z = torch.cat([lower, upper], dim=1)
         log_det = torch.sum(s1_transformed, dim=1) + \
                   torch.sum(s2_transformed, dim=1)
         return z, log_det, t2_transformed, s2_transformed
 
     def inverse(self, z):
         lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
         t2_transformed = self.t2(upper)
         s2_transformed = self.s2(upper)
         lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
         t1_transformed = self.t1(lower)
         s1_transformed = self.s1(lower)
         upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
         x = torch.cat([lower, upper], dim=1)
         log_det = torch.sum(-s1_transformed, dim=1) + \
                   torch.sum(-s2_transformed, dim=1)
         return x, log_det

class MAF(nn.Module):
    """
    Masked auto-regressive flow.
    [Papamakarios et al. 2018]
    """
    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, is_cuda=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.initial_param = nn.Parameter(torch.Tensor(2))
        self.is_cuda = is_cuda

        for i in range(1, dim):
            self.layers += [base_network(i, 2, hidden_dim, is_cuda)]
        self.reset_parameters()
        if is_cuda:
            self.cuda()

    def reset_parameters(self):
        init.uniform_(self.initial_param, -math.sqrt(0.5), math.sqrt(0.5))

    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        log_det = log_det.cuda() if self.is_cuda else log_det
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            z[:, i] = (x[:, i] - mu) / torch.exp(alpha)
            log_det -= alpha
        return z.flip(dims=(1,)), log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0])
        log_det = log_det.cuda() if self.is_cuda else log_det
        
        z = z.flip(dims=(1,))
        for i in range(self.dim):
            if i == 0:
                mu, alpha = self.initial_param[0], self.initial_param[1]
            else:
                out = self.layers[i - 1](x[:, :i])
                mu, alpha = out[:, 0], out[:, 1]
            x[:, i] = mu + torch.exp(alpha) * z[:, i]
            log_det += alpha
        
        return x, log_det


class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.
    [Durkan et al. 2019]
    """
    
    def __init__(self, dim, K = 5, B = 3, hidden_dim = 8, base_network = FCNN, is_cuda=True):
        super(NSF_AR, self).__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        
        #self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        temp = torch.Tensor(3 * K - 1)
        temp = temp.cuda() if is_cuda else temp
        self.init_param = nn.Parameter(temp)

        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim, is_cuda)]
        #self.layers = self.layers.cuda() if x.is_cuda else self.layers 
        
        self.reset_parameters()
        if is_cuda:
            self.cuda()
    
    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    
    def forward(self, x):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        log_det = log_det.cuda() if x.is_cuda else log_det 
        #temp = torch.Tensor(3 * K - 1)
        #temp = temp.cuda() if x.is_cuda else temp
        #self.init_param = nn.Parameter(temp)
        #self.init_param = self.init_param.cuda() if x.is_cuda else self.init_param
        for i in range(self.dim):
            #print("Int param", i, self.init_param.is_cuda)
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                #print(self.layers[i-1], x[:, :i].is_cuda)
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    
    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(x.shape[0])
        log_det = log_det.cuda() if x.is_cuda else log_det

        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det
