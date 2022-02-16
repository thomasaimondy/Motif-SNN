import torch
import torch.nn as nn
from function import trainingHook


class FA_wrapper(nn.Module):
    def __init__(self, module, layer_type, dim, stride=None, padding=None):
        super(FA_wrapper, self).__init__()
        self.module = module
        self.layer_type = layer_type
        self.stride = stride
        self.padding = padding
        self.output_grad = None
        self.x_shape = None

        self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim))) 
        self.reset_weights()

    def forward(self, x):
        if x.requires_grad:
            x.register_hook(self.FA_hook_pre)
            self.x_shape = x.shape
            x = self.module(x)
            x.register_hook(self.FA_hook_post)
            return x
        else:
            return self.module(x)

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)      
        self.fixed_fb_weights.requires_grad = False
    
    def FA_hook_pre(self, grad):
        if self.output_grad is not None:
            if (self.layer_type == "fc"):
                return self.output_grad.mm(self.fixed_fb_weights)   
            elif (self.layer_type == "conv"):
                return torch.nn.grad.conv2d_input(self.x_shape, self.fixed_fb_weights, self.output_grad, self.stride, self.padding)
            else:
                raise NameError("=== ERROR: layer type " + str(self.layer_type) + " is not supported in FA wrapper")
        else:
            return grad

    def FA_hook_post(self, grad):
        self.output_grad = grad
        return grad


class TrainingHook(nn.Module):
    def __init__(self, label_features, dim_hook, train_mode):
        super(TrainingHook, self).__init__()
        self.train_mode = train_mode
        assert train_mode in ["BP", "FA", "DFA", "DRTP", "sDFA", "shallow"], "=== ERROR: Unsupported hook training mode " + train_mode + "."
        
        
        if self.train_mode in ["DFA", "DRTP", "sDFA"]:
            self.fixed_fb_weights = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()
        else:
            self.fixed_fb_weights = None

    def reset_weights(self):
        torch.nn.init.kaiming_uniform_(self.fixed_fb_weights)       
        self.fixed_fb_weights.requires_grad = False

    def forward(self, input, labels, y):
        return trainingHook(input, labels, y, self.fixed_fb_weights, self.train_mode if (self.train_mode != "FA") else "BP") 

    def __repr__(self):         
        return self.__class__.__name__ + ' (' + self.train_mode + ')'
