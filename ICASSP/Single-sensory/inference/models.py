# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import function
import numpy as np
from module import FA_wrapper, TrainingHook

spike_args = {}


class NetworkBuilder(nn.Module):
    """
    This version of the network builder assumes stride-2 pooling operations.
    """

    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, train_mode, dropout,
                 conv_act, hidden_act, output_act, fc_zero_init, spike_window, device, thresh, randKill, lens, decay, xishuspike, xishuada, propnoise):     
        super(NetworkBuilder, self).__init__()  

        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window
        self.randKill = randKill
        self.label_features = label_features
        self.xishuspike = xishuspike
        self.xishuada = xishuada
        self.propnoise = propnoise
        spike_args['thresh'] = thresh
        spike_args['lens'] = lens
        spike_args['decay'] = decay

        if (train_mode == "DFA") or (train_mode == "sDFA"):
            self.y = torch.zeros(train_batch_size, label_features, device=device)   
            self.y.requires_grad = False    
        else:
            self.y = None

        topology = topology.split('_')
        self.topology = topology
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV":
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(
                        output_dim / 2)  
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(CNN_block(
                        in_channels=in_channels,
                        out_channels=int(layer[1]),
                        kernel_size=int(layer[2]),
                        stride=int(layer[3]),
                        padding=int(layer[4]),
                        bias=True,
                        activation=conv_act,
                        dim_hook=[label_features, out_channels, output_dim, output_dim],
                        label_features=label_features,
                        train_mode=train_mode,
                        batch_size=self.batch_size,
                        spike_window=self.spike_window
                    ))

                elif layer[0] == "RFC":
                    if topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])  
                        self.conv_to_fc = i
                    elif i==0:
                        input_dim = input_size if (i == 0) else output_dim
                        self.conv_to_fc = 0
                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(RFC_block(
                        in_features=input_dim,
                        out_features=output_dim,
                        bias=True,
                        activation=output_act if output_layer else hidden_act,
                        dropout=dropout,
                        dim_hook=[label_features, output_dim],
                        label_features=label_features,
                        fc_zero_init=fc_zero_init,
                        train_mode=train_mode,  
                        batch_size=train_batch_size,
                        spike_window=self.spike_window,
                        xishuspike = self.xishuspike,
                        xishuada = self.xishuada,
                        propnoise = self.propnoise
                    ))
                elif layer[0] == "FC":
                    if (i == 0):
                        
                        input_dim = input_size
                        self.conv_to_fc = 0
                        
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])  
                        self.conv_to_fc = i
                        
                    elif topology_layers[i - 1][0] == "C":
                        input_dim = output_dim//2 * int(topology_layers[i - 1][1])
                        
                        
                        self.conv_to_fc = i
                    elif topology_layers[i - 1][0] == "RNN":
                        input_dim = int(topology_layers[i - 1][3]) * 2
                        self.rnn_to_fc = i
                    else:
                        input_dim = output_dim
                        

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(FC_block(
                        in_features=input_dim,
                        out_features=output_dim,
                        bias=True,
                        activation=output_act if output_layer else hidden_act,
                        dropout=dropout,
                        dim_hook=None if output_layer else [label_features, output_dim],
                        label_features=label_features,
                        fc_zero_init=fc_zero_init,
                        train_mode=("BP" if (train_mode != "FA") else "FA") if output_layer else train_mode,
                        batch_size=train_batch_size,
                        spike_window=self.spike_window
                    ))
                elif layer[0] == "C":
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim) // 2  
                    output_dim = int((input_dim + 2*int(layer[4]) - int(layer[2]) + 1) / int(layer[3]))
                    
                    self.layers.append(C_block(
                        in_channels=in_channels,
                        out_channels=int(layer[1]),
                        kernel_size=int(layer[2]),
                        stride=int(layer[3]),
                        padding=int(layer[4]),
                        bias=True,
                        activation=conv_act,
                        dim_hook=[label_features, out_channels, output_dim],
                        label_features=label_features,
                        train_mode=train_mode,
                        batch_size=self.batch_size,
                        spike_window=self.spike_window
                    ))
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))
        self.vote = nn.Linear(in_features=int(layer[1]), out_features=label_features, bias=True)

    def forward(self, input, labels, spike_window_1014, onoffnoise, dataset):
        
        result = None 

        for step in range(spike_window_1014):
            if self.topology[0] == 'C':
                x = input[:,:,:,step]
            elif self.topology[0] == 'CONV': 
                if dataset=='tidigits' or dataset == 'MNIST':
                    x = input.float().unsqueeze(1)
                    
                    prop = self.propnoise  

                    x = x.cuda() + (((torch.rand(x.size()) - 0.5) * 0.2) * prop).cuda()  

                    x = x > torch.rand(x.size()).float().cuda() * self.randKill
                    if labels is not None:
                        label = labels[:, step, :]
                    else:
                        label = labels
                
            elif self.topology[0] == 'RFC':
                if dataset=='timit':
                    x = input[:,step,:]    
                    if labels is not None:
                        label = labels[:,step,:]
                    else:
                        label = labels
                if dataset=='tidigits' or dataset == 'MNIST':
                    x = input[:, step, :]
                    if labels is not None:
                        label = labels[:, step, :]
                    else:
                        label = labels
                if dataset=='dvsgesture':
                    x = input[:, step, :] 
                    if labels is not None:
                        label = labels[:, step, :]
                    else:
                        label = labels
            elif self.topology[0] == 'FC':
                x = input[:,step,:]   
                label = labels[:,step,:]
            x = x.float()


            for i in range(len(self.layers)):
                if i == self.conv_to_fc:
                    x = x.reshape(x.size(0), -1)
                x = self.layers[i](x, label, self.y, spike_window_1014, onoffnoise)
            x = self.vote(x)
            x = torch.sigmoid(x)
        
            if result is None: 
                result = x
            else:
                result = torch.cat((result, x), 1)

            if result.requires_grad and (self.y is not None):
                self.y.data.copy_(x.data)  
            
        result = result.view(result.shape[0],spike_window_1014, x.shape[-1])

        return result

class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(spike_args['thresh']).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - spike_args['thresh']) < spike_args['lens']
        return grad_input * temp.float()

act_fun = ActFun.apply

class RFC_block(nn.Module): 
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init,
                 train_mode, batch_size, spike_window, xishuspike, xishuada, propnoise):
        super(RFC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.spike_window = spike_window
        self.dropout = dropout
        self.xishuspike = xishuspike
        self.xishuada = xishuada
        self.propnoise = propnoise
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.rec = nn.Linear(in_features=out_features, out_features=out_features)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook_ff = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        self.hook_r = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        self.adapThreshold = None 
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        cmask = torch.load('/home/user/zuoruichen/code/Motif/mask/CASNNTIDbp4-4/mask_before_sigmoid_300_50.torch')

        cmask = torch.where(cmask > 0, torch.ones_like(cmask), torch.zeros_like(cmask)) 
        self.rec_mask = torch.nn.Parameter(cmask).cuda()

    def mem_update(self, x, labels, y, onoffnoise):
        xishuspike, xishuada = self.xishuspike, self.xishuada
        self.adapThreshold = 0.9* self.adapThreshold +  xishuspike*self.spike   
        mem_ff = self.mem * spike_args['decay'] * (1. - self.spike) + self.fc(x) - self.adapThreshold*xishuada 
        spike_ff = self.hook_ff(act_fun(mem_ff), labels, y)
        if self.rec:
            self.rec.weight.data = self.rec.weight * self.rec_mask 
            mem_r = self.rec(self.spike)
            spike_r = self.hook_ff(act_fun(mem_r), labels, y)
        self.mem = mem_ff + mem_r
        self.spike = spike_ff + spike_r

    def forward(self, x, labels, y, spike_window_1014, onoffnoise):
        
        if self.time_counter == 0:
            
            self.rec.weight.data = self.rec.weight * self.rec_mask
            self.mem = torch.zeros((x.shape[0], self.out_features)).cuda()
            self.spike = torch.zeros((x.shape[0], self.out_features)).cuda()
            self.sumspike = torch.zeros((x.shape[0], self.out_features)).cuda()
            self.adapThreshold = torch.zeros((x.shape[0], self.out_features)).cuda()

        if False:
            x = self.drop(x)

        self.time_counter += 1
        self.mem_update(x, labels, y, onoffnoise)
        self.sumspike += self.spike
        
        if self.time_counter == spike_window_1014:
            self.time_counter = 0
        return self.spike

class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook,label_features, train_mode, batch_size, spike_window):
        super(CNN_block, self).__init__()
        self.spike_window = spike_window
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        
        if train_mode == 'FA':
            self.conv = FA_wrapper(module=self.conv, layer_type='conv', dim=self.conv.weight.shape, stride=stride,
                                   padding=padding)
        self.act = Activation(activation)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels

    def mem_update(self, ops, x, mem, spike, lateral=None):
        a = ops(x)
        mem = mem * spike_args['decay'] * (1. - spike) + ops(x)
        

        if lateral:
            mem += lateral(spike)
        spike = act_fun(mem)
        return mem, spike
    
    def forward(self, x, labels, y, spike_window_1014, onoffnoise):
        
        if self.time_counter == 0:
            self.mem = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).cuda()
            self.spike = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).cuda()
            self.sumspike = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).cuda()

        self.time_counter += 1
        self.mem, self.spike = self.mem_update(self.conv, x, self.mem, self.spike)

        x = self.hook(self.spike, labels, y)

        x = self.pool(x)

        if self.time_counter == spike_window_1014:
            self.time_counter = 0

        return x

class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, label_features, fc_zero_init,
                 train_mode, batch_size, spike_window):
        super(FC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.spike_window = spike_window
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if train_mode == 'FA':
            self.fc = FA_wrapper(module=self.fc, layer_type='fc', dim=self.fc.weight.shape)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.hook = TrainingHook(label_features=label_features, dim_hook=dim_hook, train_mode=train_mode)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0

    def mem_update(self, ops, x, mem, spike, lateral=None):
        a=ops(x)
        mem = mem * spike_args['decay'] * (1. - spike) + ops(x)
        if lateral:
            mem += lateral(spike)
        spike = act_fun(mem)
        return mem, spike

    def forward(self, x, labels, y, spike_windows_1014, onoffnoise):
        
        if self.time_counter == 0:
            self.mem = torch.zeros((x.shape[0], self.out_features)).cuda()
            self.spike = torch.zeros((x.shape[0], self.out_features)).cuda()
            self.sumspike = torch.zeros((x.shape[0], self.out_features)).cuda()

        if False:
            x = self.drop(x)

        self.time_counter += 1
        self.mem, self.spike = self.mem_update(self.fc, x, self.mem, self.spike)
        self.sumspike += self.spike

        x = self.hook(self.spike, labels, y)
        if self.time_counter == spike_windows_1014:
            self.time_counter = 0
        return x

class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)
