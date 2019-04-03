import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.utils import data
from torch.autograd import Function

#Dataset class
class Dataset(data.Dataset):
    def __init__(self, X, y, alpha=None, weights=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.a = None
        self.w = None
        if type(alpha) != type(None):
            self.a = torch.from_numpy(alpha).float()
        if type(weights) != type(None):
            self.w = torch.from_numpy(weights).float()
    
    def __len__(self):
        return self.y.shape[0]
      
    def __getitem__(self, index):
        if type(self.a) != type(None) and type(self.w) != type(None):
            return self.X[index, :], self.y[index], self.a[index], self.w[index]
        if type(self.a) != type(None) and type(self.w) == type(None):
            return self.X[index, :], self.y[index], self.a[index]
        if type(self.a) == type(None) and type(self.w) != type(None):
            exit(1)
        return self.X[index, :], self.y[index]

#the calibrated logistic loss
class C_BCE(Function):
  
    @staticmethod
    def forward(ctx, input, target, alpha):
        ctx.save_for_backward(input, target, alpha)
        return -(target*(1-alpha)*torch.log(input) + (1-target)*alpha*torch.log(1-input)).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, alpha = ctx.saved_tensors
        grad_input, grad_target, grad_alpha = None, None, None
        grad_input = -target*torch.div(1-alpha, input) + (1-target)*torch.div(alpha, 1-input)
        grad_input *= grad_output
        return grad_input, None, None

#check the gradient
'''
from torch.autograd import gradcheck
target = torch.randn(50, 1, dtype=torch.double)
target = torch.where(target > 0, torch.ones(target.size()), torch.zeros(target.size())).double()
target.requires_grad = False
alpha = torch.randn(50, 1, dtype=torch.double)
alpha = torch.where(target > 0, torch.ones(target.size())/2.0 + 1, torch.ones(target.size()) - 4).double()
alpha.requires_grad = False
input = (torch.exp(torch.randn(50, 1, dtype=torch.double, requires_grad=True)).uniform_(),
         target,
         alpha
         )
test = gradcheck(C_BCE.apply, input, eps = 1e-7, atol=1e-4)
print(test)
'''

class C_hinge(Function):
  
    @staticmethod
    def forward(ctx, input, target, alpha):
        zeros = torch.zeros(input.size(), dtype=torch.double)
        ones = torch.ones(input.size(), dtype=torch.double)
        mask = torch.where(target == 1, 1-alpha, alpha).double()
        ctx.save_for_backward(input, target, alpha, zeros, mask)
        return (mask * torch.where(input*target < 1, ones-target*input, zeros)).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, alpha, zeros, mask = ctx.saved_tensors
        grad_input, grad_target, grad_alpha = None, None, None
        #grad_input = mask * torch.where(input*target < 1, -target, zeros)
        grad_input = mask * (input*target < 1).double() * (-target)
        grad_input *= grad_output
        return grad_input, None, None

class C_logistic(Function):
  
    @staticmethod
    def forward(ctx, input, target, alpha):
        zeros = torch.zeros(input.size(), dtype=torch.double)
        #ones = torch.ones(input.size(), dtype=torch.double)
        mask = torch.where(target == 1, 1-alpha, alpha).double()
        ctx.save_for_backward(input, target, alpha, mask)
        return (mask * torch.log(1 + torch.exp(-target*input))).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, alpha, mask = ctx.saved_tensors
        grad_input, grad_target, grad_alpha = None, None, None
        #grad_input = mask * (input*target < 1).double() * (-target)
        grad_input = mask * torch.exp(-target*input) * (-target) * \
                     torch.div(1, 1+torch.exp(-target*input))
        grad_input *= grad_output
        return grad_input, None, None

class C_BCE_weighted(Function):
  
    @staticmethod
    def forward(ctx, input, target, alpha, weights=None):
        if type(weights) == type(None):
            weights = torch.ones(target.size(), dtype=torch.double)
        ctx.save_for_backward(input, target, alpha, weights)
        return (-weights*(target*(1-alpha)*torch.log(input) + (1-target)*alpha*torch.log(1-input))).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, alpha, weights = ctx.saved_tensors
        grad_input, grad_target, grad_alpha, grad_weights = None, None, None, None
        grad_input = weights*(-target*torch.div(1-alpha, input) + (1-target)*torch.div(alpha, 1-input))
        grad_input *= grad_output
        return grad_input, grad_target, grad_alpha, grad_weights

class C_hinge_weighted(Function):
  
    @staticmethod
    def forward(ctx, input, target, alpha, weights):
        zeros = torch.zeros(input.size(), dtype=torch.double)
        ones = torch.ones(input.size(), dtype=torch.double)
        #mask = torch.where(target == 1, 1-alpha, alpha).double()
        mask = ((target == 1).double()*(1-alpha) + (target != 1).double()*alpha).double()
        ctx.save_for_backward(input, target, alpha, mask, weights)
        #return (weights * mask * torch.where(input*target < 1, ones-target*input, zeros)).sum()
        return (weights * mask * (input*target < 1).double() * (1.0-target*input)).sum()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, target, alpha, mask, weights = ctx.saved_tensors
        grad_input, grad_target, grad_alpha, weights_grad = None, None, None, None
        #grad_input = mask * torch.where(input*target < 1, -target, zeros)
        grad_input = weights * mask * (input*target < 1).double() * (-target)
        grad_input *= grad_output
        return grad_input, grad_target, grad_alpha, weights_grad