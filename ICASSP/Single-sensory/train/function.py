# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function
from numpy import prod

class HookFunction(Function):
    @staticmethod
    def forward(ctx, input, labels, y, fixed_fb_weights, train_mode):
        if train_mode in ["DFA", "sDFA", "DRTP"]:
            ctx.save_for_backward(input, labels, y, fixed_fb_weights)       
        ctx.in1 = train_mode
        return input

    @staticmethod
    def backward(ctx, grad_output):     
        train_mode          = ctx.in1
        if train_mode == "BP":
            return grad_output, None, None, None, None
        elif train_mode == "shallow":
            grad_output.data.zero_()
            return grad_output, None, None, None, None
        
        input, labels, y, fixed_fb_weights = ctx.saved_variables
        if train_mode == "DFA":
            grad_output_est = (y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)     
        elif train_mode == "sDFA":
            grad_output_est = torch.sign(y-labels).mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
        elif train_mode == "DRTP":
            grad_output_est = labels.mm(fixed_fb_weights.view(-1,prod(fixed_fb_weights.shape[1:]))).view(grad_output.shape)
            
        else:
            raise NameError("=== ERROR: training mode " + str(train_mode) + " not supported")

        return grad_output_est, None, None, None, None

trainingHook = HookFunction.apply
