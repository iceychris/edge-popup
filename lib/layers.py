import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from IPython.core.debugger import set_trace

K = 0.3


def signed_kaiming_constant_(tensor, a=0, mode='fan_in', nonlinearity='relu', k=1.):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = (gain / math.sqrt(fan))
    # scale by (1/sqrt(k))
    std *= (1 / math.sqrt(k))
    with torch.no_grad():
        return tensor.uniform_(-std, std)


class GetSubnet(autograd.Function):
    
    @staticmethod
    def forward(ctx, scores, k):
        
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1-k) * scores.numel())
        
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        
        return out
    
    @staticmethod
    def backward(ctx, grad):
        
        # send the gradient g straight-through on the backward pass.
        return grad, None


class LinearSubnet(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, k=K, init=signed_kaiming_constant_, **kwargs):
        super(LinearSubnet, self).__init__(in_features, out_features, bias if isinstance(bias, bool) else True, **kwargs)
        self.k = k
        self.popup_scores = nn.Parameter(torch.randn(*self.weight.shape))

        # init weights
        init(self.weight, k=k)

        # disable grad for the original parameters
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)

        # Use only the subnetwork in the forward pass.
        w = self.weight * adj
        x = F.linear(x, w)
        return x


class Conv2dSubnet(nn.Conv2d):
    
    def __init__(self, *args, k=K, init=signed_kaiming_constant_, **kwargs):
        super(Conv2dSubnet, self).__init__(*args, **kwargs)
        self.k = k
        self.popup_scores = nn.Parameter(torch.randn(*self.weight.shape))

        # init weights
        init(self.weight, k=k)

        # disable grad for the original parameters
        self.weight.requires_grad_(False)
        if self.bias:
            self.bias.requires_grad_(False)
    
    # self.k is the % of weights remaining, a real number in [0,1]
    # self.popup_scores is a Parameter which has the same shape as self.weight
    # Gradients to self.weight, self.bias have been turned off.
    def forward(self, x):
        # Get the subnetwork by sorting the scores.
        adj = GetSubnet.apply(self.popup_scores.abs(), self.k)
        
        # Use only the subnetwork in the forward pass.
        w = self.weight * adj
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
