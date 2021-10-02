# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import models
from tqdm import tqdm
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os
import sys
from torch import set_default_tensor_type
from collections import Counter
import numpy as np

def train(args, device, train_loader, traintest_loader, test_loader):
    torch.manual_seed(args.seed) 

    for trial in range(1,args.trials+1):
        

        model = models.NetworkBuilder(args.topology, input_size=args.input_size, input_channels=args.input_channels, label_features=args.label_features, train_batch_size=args.batch_size, train_mode=args.train_mode, dropout=args.dropout, conv_act=args.conv_act, hidden_act=args.hidden_act, output_act=args.output_act, fc_zero_init=args.fc_zero_init, spike_window=args.spike_window,  device=device, thresh=args.thresh, randKill=args.randKill, lens=args.lens, decay=args.decay, xishuspike=args.xishuspike, xishuada=args.xishuada, propnoise=args.propnoise)

        if args.cuda:
            model.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=False)
        elif args.optimizer == 'NAG':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.L2)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lrstep, gamma=args.lrgama)
        elif args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NameError("=== ERROR: optimizer " + str(args.optimizer) + " not supported")

        if args.loss == 'MSE':
            loss = (F.mse_loss, (lambda l : l))
        elif args.loss == 'BCE':
            loss = (F.binary_cross_entropy, (lambda l : l))
        
        
        elif args.loss == 'CE':
            
            loss = (nn.CrossEntropyLoss(), (lambda l : l))
        else:
            raise NameError("=== ERROR: loss " + str(args.loss) + " not supported")

        print("\n\n=== Starting model training with %d epochs:\n" % (args.epochs,))

        filepath = 'model/' + args.codename.split('-')[0] + '/' + args.codename
        
        if True:
            checkpoint = torch.load(os.path.join('/home/jiashuncheng/code/muticasnn/model',args.path)+'/model.pth')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            start_epoch = checkpoint['epoch']+1
            print('加载 epoch {} 成功！继续从epoch {} 开始测试。'.format(start_epoch-1,start_epoch))
        else:
            start_epoch = 1
            print('无保存模型，将从头开始训练！')

        for epoch in range(start_epoch, args.epochs + 1):
            
            print("\nSummary of epoch %d:" % (epoch))
            
            test_epoch(args, model, device, test_loader, loss, 'Test',epoch, onoffnoise=args.test_noise)
           
            np.save(str(args.Mode)+'_'+args.path.split('/')[1]+'hidden_output.npy',np.array(model.layers[1].hidden_out))
            break


def train_epoch(args, model, device, train_loader, optimizer, loss, onoffnoise):
    model.train()

    if args.freeze_conv_layers:
        for i in range(model.conv_to_fc):
            for param in model.layers[i].conv.parameters():
                param.requires_grad = False
        for param in model.layers[1].fc.parameters():
            param.requires_grad = False
        for param in model.layers[1].rec.parameters():
            param.requires_grad = False

    if args.dataset == 'MNISTtidigits':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            
            data1 = data[0] 
            data2 = data[1] 
            label = label[0]
            data1, data2, label = data1.to(device), data2.to(device),label.to(device)
            data1 = data1.squeeze(1)
            data2 = data2.squeeze(1)
            label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)    
            label = label.unsqueeze(1).repeat(1,args.spike_window,1)    

            max_len = args.spike_window 
            optimizer.zero_grad()
            output = model([data1, data2], label, max_len, onoffnoise,args.dataset) 
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()

    if args.dataset == 'dvsgesture':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            data = data.squeeze(1).view(data.shape[0],-1,data.shape[-1]*data.shape[-2])
            label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)    
            label = label.unsqueeze(1).repeat(1,data.shape[-2],1)

            max_len = data.shape[-2]
            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise, args.dataset) 
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()

    if args.dataset == 'tidigits' or args.dataset=='MNIST':
        for batch_idx, (data, label) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            data = data.squeeze(1)
            label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)    
            label = label.unsqueeze(1).repeat(1,data.shape[-2],1)

            max_len = data.shape[-1]
            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise,args.dataset) 
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()
    if args.dataset == 'timit':
        for batch_idx, (data, label, seq_len) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            
            max_len = seq_len.max()
            data = data[:, :max_len, :]
            label = label[:, :max_len] 

            
            label = torch.zeros(label.shape[0], label.shape[1], args.label_features, device=device).scatter_(2, label.unsqueeze(2).long(), 1.0)    

            optimizer.zero_grad()
            output = model(data, label, max_len, onoffnoise,args.dataset) 
            loss_val = loss[0](output, loss[1](label))
            loss_val.backward(retain_graph=True)
            optimizer.step()

    if args.dataset == 'others':
        for batch_idx, (data, label, seq_len) in enumerate(tqdm(train_loader)):
            data, label = data.to(device), label.to(device)
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(1).long(), 1.0)
            optimizer.zero_grad()
            output = model(data, targets)
            loss_val = loss[0](output, loss[1](targets))
            loss_val.backward(retain_graph = True)
            optimizer.step()

def writefile(args, file):
    filepath = 'output/'+args.codename.split('-')[0]+'/'+args.codename
    filetestloss = open(filepath + file, 'a')
    return filetestloss

def test_epoch(args, model, device, test_loader, loss, phase, epoch, onoffnoise):
    model.eval()

    test_loss, correct = 0, 0
    out = []
    
    len_dataset = len(test_loader.dataset)
    
    with torch.no_grad():
        if args.dataset == 'MNISTtidigits':
            for batch_idx, (data, label) in enumerate(test_loader):
                data1 = data[0]
                data2 = data[1]
                label = label[0]
                data1, data2, label = data1.to(device), data2.to(device), label.to(device)
                data1 = data1.squeeze(1)
                data2 = data2.squeeze(1)
                label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(
                    1).long(), 1.0)  
                label = label.unsqueeze(1).repeat(1, data1.shape[-2], 1)

                max_len = data1.shape[-1]

                output = model([data1, data2], None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                out.extend(phn_prediction.reshape(-1).cpu().numpy().tolist())
                targets = torch.argmax(label, dim=2)

                mach1, number1 = (targets == phn_prediction).float().sum(), targets.numel()
                correct += (mach1 / number1).item()

        if args.dataset == 'dvsgesture':
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = data.squeeze(1).view(data.shape[0],-1,data.shape[-1]*data.shape[-2])
                label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(
                    1).long(), 1.0)  
                label = label.unsqueeze(1).repeat(1, data.shape[-2], 1)

                max_len = data.shape[-2]

                output = model(data, None, max_len, onoffnoise, args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label, dim=2)

                mach1, number1 = (targets == phn_prediction).float().sum(), targets.numel()
                correct += (mach1 / number1).item()

        if args.dataset == 'tidigits' or args.dataset=='MNIST':
            for batch_idx, (data, label) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                data = data.squeeze(1)
                label = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1, label.unsqueeze(
                    1).long(), 1.0)  
                label = label.unsqueeze(1).repeat(1, data.shape[-2], 1)

                max_len = data.shape[-1]

                output = model(data, None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label, dim=2)

                mach1, number1 = (targets == phn_prediction).float().sum(), targets.numel()
                correct += (mach1 / number1).item()

        if args.dataset == 'timit':
            for batch_idx, (data, label, seq_len) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                max_len = seq_len.max()
                data = data[:, :max_len, :]
                label = label[:, :max_len]  

                label = torch.zeros(label.shape[0], label.shape[1], args.label_features, device=device).scatter_(2, label.unsqueeze(2).long(),1.0)  

                output = model(data, None, max_len, onoffnoise,args.dataset)

                test_loss += loss[0](output, loss[1](label)).item()
                phn_prediction = torch.argmax(output, dim=2)
                targets = torch.argmax(label,dim=2)

                mach1, number1 = 0, 0
                for ic in range(phn_prediction.shape[0]):
                    mach1 += (phn_prediction[ic,:seq_len[ic]] == targets[ic,:seq_len[ic]]).float().sum()
                    number1 += phn_prediction[ic,:seq_len[ic]].numel()

                correct += (mach1/number1).item()

        if args.dataset == 'others':
            targets = torch.zeros(label.shape[0], args.label_features, device=device).scatter_(1,label.unsqueeze(1).long(), 1.0)

            output = model(data, None)

            test_loss += loss[0](output, loss[1](targets), reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred).long()).sum().item()
            

    loss = test_loss / (batch_idx+1)
    print(Counter(out))
    if True:
        acc = correct / (batch_idx+1)
        print("\t[%5sing set] Loss: %6f, Accuracy: %1.6f" % (phase, loss, acc))

        filetestloss = writefile(args, '/testloss.txt')
        filetestacc = writefile(args, '/testacc.txt')
        filetrainloss = writefile(args, '/trainloss.txt')
        filetrainacc = writefile(args, '/trainacc.txt')

        if phase == 'Train':
            filetrainloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetrainacc.write(str(epoch) + ' ' + str(acc) + '\n')
        if phase == 'Test':
            filetestloss.write(str(epoch) + ' ' + str(loss) + '\n')
            filetestacc.write(str(epoch) + ' ' + str(acc) + '\n')
    else:
        print("\t[%5sing set] Loss: %6f" % (phase, loss))
