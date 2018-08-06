# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 22:05:37 2018

@author: User
"""

import numpy as np
import data.loaders as dl
from data.directories import T_list
from argparse import ArgumentParser

parser = ArgumentParser()
## Data parameters
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-iT', type=int, default=0, help='temperature index')
parser.add_argument('-L', type=int, default=16, help='configuration size')
parser.add_argument('-nTR', type=int, default=10000, help='training samples')
parser.add_argument('-nTE', type=int, default=10000, help='test samples')
parser.add_argument('-nVAL', type=int, default=5000, help='free energy validation samples')
parser.add_argument('-VER', type=int, default=1, help='version for name')

## Training parameters
parser.add_argument('-GPU', type=float, default=0.3, help='GPU fraction')
parser.add_argument('-EP', type=int, default=10000, help='number of epochs')
parser.add_argument('-BS', type=int, default=50, help='batch size')
parser.add_argument('-BSC', type=int, default=5000, help='calculation batch size (for memory)')

## Early stopping options
parser.add_argument('-ES', type=bool, default=True, help='use early stopping')
parser.add_argument('-ESPAT', type=int, default=300, help='early stopping patience')
parser.add_argument('-ESTHR', type=float, default=3e-4, help='early stopping threshold')

## Learning rate options
parser.add_argument('-LR', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-LRF', type=float, default=0.001, help='final learning rate')
parser.add_argument('-LREP', type=int, default=1000, help='epoch to start learning rate decay')

## Average weights during training
parser.add_argument('-WAVEP', type=int, default=None, help='epoch to start weight averaging')
parser.add_argument('-WAVNR', type=int, default=5, help='number of weights to average')
parser.add_argument('-WAVST', type=int, default=5, help='step to make the updates')
parser.add_argument('-SAVEWL', type=bool, default=False, help='save weight list')

## RBM convolutions parameters
parser.add_argument('-Nw', nargs='+', type=int, default=None, help='filter size')
parser.add_argument('-K', nargs='+', type=int, default=None, help='hidden groups')

## RBM Gibbs sampling parameters
parser.add_argument('-GBTR', type=int, default=20, help='Gibbs for CD')
parser.add_argument('-GBMSE', type=int, default=1, help='Gibbs for testing')
parser.add_argument('-GBTE', type=int, default=5, help='Gibbs for testing')

## Message parameters
parser.add_argument('-MSG', type=int, default=2, help='epoch message')
parser.add_argument('-MSGT', type=int, default=10, help='observables test message')


def main(args):   
    ## Decide if deep or not
    assert len(args.K) == len(args.Nw)
    if len(args.K) > 1:
        from networks.trainer_deep import Trainer
        args.WAVEP = None ##!!! we have not fixed weight averaging for deep model yet!
        
    else:
        from networks.trainer import Trainer
        # Make Nw and K integers
        args.Nw, args.K = args.Nw[0], args.K[0]
    
    ## Load data
    train_data = dl.add_index(dl.read_file(L=args.L, n_samples=10000, train=True))
    val_data = dl.add_index(dl.read_file(L=args.L, n_samples=10000, train=False))
    
    for (iT, T) in enumerate(T_list):
        ## Prepare RBM
        args.iT = iT
        rbm = Trainer(args)
        
        print('\nTemperature: %.4f  -  Critical: %s'%(T, str(args.CR)))
        if isinstance(args.K, int):
            print('RBM with %d visible units and %d hidden units.'%(rbm.Nv**2, rbm.Nh**2*rbm.K))
            print('Number of weight parameters: %d.\n'%(rbm.Nw**2*rbm.K))
            
        else:
            print('Created RBM with %d visible units.'%(rbm.Nv**2))
            print(args.K)
            print('\n')
        
        rbm.prepare_training()
    
        ## Set up directory for saving       
        save_dir = 'Trained_Models/MultipleTemps/%s'%(rbm.name)
        dl.create_directory(save_dir)
        save_dir += '/T%.4f'%T
        dl.create_directory(save_dir)
        
        ## Train and save
        rbm.fit_and_save(train_data=dl.temp_partition(train_data, iT), 
                         val_data=dl.temp_partition(val_data, iT)[:args.nVAL], 
                         directory=save_dir)
        
        print('Temperature %d / %d done!'%(iT+1, len(T_list)))
    
    return
    
main(parser.parse_args())

