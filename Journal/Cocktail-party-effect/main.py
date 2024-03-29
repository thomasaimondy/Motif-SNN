# -*- coding: utf-8 -*-
import argparse
import train
import setup
import os
import utils
import time
import torch
import numpy as np

def filedel(filepath):
    for i in ['/testloss.txt','/testacc.txt','/trainloss.txt','/trainacc.txt']:
        try:
            os.remove(filepath+i)
        except:
            pass
def mkd(args):
    if not os.path.isdir('output/' + args.codename):
        os.makedirs('output/' + args.codename)
    if not os.path.isdir('model/' + args.codename):
        os.makedirs('model/' + args.codename)

def main():
    parser = argparse.ArgumentParser(description='Training fully-connected and convolutional networks using backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), and direct random target projection (DRTP)')
    # General
    parser.add_argument('--cpu', action='store_true', default=False, help='Disable CUDA and run on CPU.')
    # Dataset
    parser.add_argument('--dataset', type=str, choices = ['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug', 'tidigits', 'MNISTtidigits', 'dvsgesture','timit'], default='MNISTtidigits', help='Choice of the dataset1: synthetic regression (regression_synth), synthetic classification (classification_synth), MNIST (MNIST), MNISTtidigits (MNIST+tidigits), CIFAR-10 (CIFAR10), CIFAR-10 with data augmentation (CIFAR10aug). Synthetic datasets must have been generated previously with synth_dataset_gen.py. Default: MNIST.')
    # Training
    parser.add_argument('--train-mode', choices = ['BP','FA','DFA','DRTP','sDFA','shallow'], default='DRTP', help='Choice of the training algorithm - backpropagation (BP), feedback alignment (FA), direct feedback alignment (DFA), direct random target propagation (DRTP), error-sign-based DFA (sDFA), shallow learning with all layers freezed but the last one that is BP-trained (shallow). Default: DRTP.')#BP DRTP
    parser.add_argument('--optimizer', choices = ['SGD', 'NAG', 'Adam', 'RMSprop'], default='Adam', help='Choice of the optimizer - stochastic gradient descent with 0.9 momentum (SGD), SGD with 0.9 momentum and Nesterov-accelerated gradients (NAG), Adam (Adam), and RMSprop (RMSprop). Default: NAG.')
    parser.add_argument('--loss', choices = ['MSE', 'BCE', 'CE'], default='MSE', help='Choice of loss function - mean squared error (MSE), binary cross entropy (BCE), cross entropy (CE, which already contains a logsoftmax activation function). Default: BCE.')#MSE BCE
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False, help='Disable training of convolutional layers and keeps the weights at their initialized values.')
    parser.add_argument('--fc-zero-init', action='store_true', default=False, help='Initializes fully-connected weights to zero instead of the default He uniform initialization.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout probability (applied only to fully-connected layers). Default: 0.')#可以试一个0.05
    parser.add_argument('--trials', type=int, default=1, help='Number of training trials Default: 1.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs Default: 100.')
    parser.add_argument('--batch-size', type=int, default=50, help='Input batch size for training. Default: 100.')
    parser.add_argument('--test-batch-size', type=int, default=50, help='Input batch size for testing Default: 100.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate. Default: 1e-4.')
    # Network
    #CONV_32_5_1_2_FC_1000_FC_10 C_100_3_1_1_FC_1000_FC_10 RNN_200_39_200_1_FC_39
    parser.add_argument('--topology', type=str, default='INTE_28_5_1_2_FC_200', help='Choice of network topology. Format for convolutional layers: CONV_{output channels}_{kernel size}_{stride}_{padding}. Format for fully-connected layers: FC_{output units}. Format for RNN layers: RNN_{input_size}_{hidden_size}_{num_layers}')#CONV_8_3_1_1_FC_100_FC_10 'FC_30_FC_10';;'CONV_32_5_1_2_FC_1000_FC_10'
    parser.add_argument('--spike_window', type=int, default=20, help='The time clock for neurons. Default: 20.')#10,20,30
    parser.add_argument('--conv-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='sigmoid', help='Type of activation for the convolutional layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--hidden-act', type=str, choices = {'tanh', 'sigmoid', 'relu'}, default='tanh', help='Type of activation for the fully-connected hidden layers - Tanh (tanh), Sigmoid (sigmoid), ReLU (relu). Default: tanh.')
    parser.add_argument('--output-act', type=str, choices = {'sigmoid', 'tanh', 'none'}, default='sigmoid', help='Type of activation for the network output layer - Sigmoid (sigmoid), Tanh (tanh), none (none). Default: sigmoid.')
    parser.add_argument('--thresh', type=float,default=0.5)     # MNIST
    parser.add_argument('--randKill', type=float,default=0.1)
    parser.add_argument('--lens', type=float,default=0.5)
    parser.add_argument('--decay', type=float, default=0.2)
    parser.add_argument('--codename', type=str, default='TTEST')
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--L2', type=bool, default=1e-5, help='l2 regularization')
    parser.add_argument('--lrstep', type=int, default=10000, help='lr step-size')
    parser.add_argument('--lrgama', type=float, default=1.0, help='lr gama')
    parser.add_argument('--xishuspike', type=float, default=0.1, help='ratio spike')
    parser.add_argument('--xishuada', type=float, default=0, help='ratio ada')
    parser.add_argument('--propnoise', type=float, default=0, help='prop noise')
    parser.add_argument('--propeight', type=float, default=0, help='prop noise')
    parser.add_argument('--train-noise', type=str, default='off', help='(default=%(default)s)')
    parser.add_argument('--test-noise', type=str, default='on', help='(default=%(default)s)')
    parser.add_argument('--gpu',type=str,default='3',help='(default=%(default)s)')
    parser.add_argument('--seed', type=int, default=42, help='(default=%(default)d)')
    parser.add_argument('--load-model-path', type=str, default='/home/jiashuncheng/code/CASNN/zuoruichen_model')
    parser.add_argument('--only-inference', action='store_true')
    parser.add_argument('--mode', type=int, default=None, help='(default=%(default)d)')
    parser.add_argument('--path', type=str, default=None, help='(default=%(default)d)')
    parser.add_argument('--cmask', type=str, default='VorA', help='(default=%(default)s)')
    parser.add_argument('--data-mode', type=str, default='TM', help='TM,TT,MM')
    parser.add_argument('--spike-type', type=int, default=1)
    parser.add_argument('--other_method', type=str, default=None)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    timeclock = time.strftime("%Y%m%d%Hh%Mm%Ss", time.localtime())
    if args.path is None:
        args.codename = timeclock + args.codename + '_' +str(args.seed) + '_' + str(int(10*args.propnoise))
        mkd(args)
        filepath = 'output/'+args.codename
        file = open(filepath+'/para.txt','w')
        file.write('pid:'+str(os.getpid())+'\n')
        file.write(str(vars(args)).replace(',','\n'))
        file.close()
        if not args.cont:
            filedel(filepath)
    else:
        args.codename = args.path
    print(args.codename)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    utils.args = args
    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)

    torch.set_num_threads(1)
    train.train(args, device, train_loader, traintest_loader, test_loader)


if __name__ == '__main__':
    main()
