# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:39:25 2018

@author: Stavros
"""

import numpy as np
from trainer import Trainer
from data.loaders import add_index, create_directory
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
parser.add_argument('-EP', type=int, default=500, help='number of epochs')
parser.add_argument('-BS', type=int, default=50, help='batch size')
parser.add_argument('-BSC', type=int, default=5000, help='calculation batch size (for memory)')

## Learning rate options
parser.add_argument('-LR', type=float, default=0.001, help='initial learning rate')
#parser.add_argument('-LRD', type=bool, default=False, help='learning rate decay')
# only linear lR decay fixed currently
parser.add_argument('-LRF', type=float, default=0.001, help='final learning rate')
parser.add_argument('-LREP', type=int, default=500, help='epoch to start learning rate decay')


## Average weights during training
parser.add_argument('-WAVEP', type=int, default=None, help='epoch to start weight averaging')
parser.add_argument('-WAVNR', type=int, default=5, help='number of weights to average')
parser.add_argument('-WAVST', type=int, default=5, help='step to make the updates')

## RBM parameters
parser.add_argument('-Nw', type=int, default=5, help='filter size')
parser.add_argument('-K', type=int, default=20, help='hidden groups')
parser.add_argument('-GBTR', type=int, default=20, help='Gibbs for CD')
parser.add_argument('-GBMSE', type=int, default=1, help='Gibbs for testing')
parser.add_argument('-GBTE', type=int, default=5, help='Gibbs for testing')

## Message parameters
parser.add_argument('-MSG', type=int, default=2, help='epoch message')
parser.add_argument('-MSGT', type=int, default=10, help='observables test message')


def main(args):    
    if args.CR:
        from data.loaders import read_file_critical
        T = 2 / np.log(1 + np.sqrt(2))

        save_dir = 'Weights/Critical'
        
        data = add_index(read_file_critical(L=args.L, n_samples=100000))
        train_data = data[:args.nTR]
        val_data = data[args.nTR : args.nTR + args.nVAL]

    else:
        from data.loaders import read_file, temp_partition
        from data.directories import T_list
        T = T_list[args.iT]
        
        save_dir = 'Weights/T%.4f'%T_list[args.iT]
        
        train_data = add_index(temp_partition(read_file(L=args.L, n_samples=10000, train=True), args.iT))
        val_data = add_index(temp_partition(read_file(L=args.L, n_samples=10000, train=False), args.iT))
        
    create_directory(save_dir)
    
    rbm = Trainer(args)
    print('\nTemperature: %.4f  -  Critical: %s'%(T, str(args.CR)))
    print('RBM with %d visible units and %d hidden units.'%(rbm.Nv**2, rbm.Nh**2*rbm.K))
    print('Number of weight parameters: %d.\n'%(rbm.Nw**2*rbm.K))
    rbm.prepare_training()
    W, hb, vb, metrics = rbm.fit(train_data=train_data, val_data=val_data, T=T)
    
    # Save weights and biases list
    save_dir += '/CRBM_EP%dBS%dLR%.5f_VER%d'%(args.EP, args.BS, args.LR, args.VER)
    create_directory(save_dir)
    np.save('%s/weights.npy'%save_dir, np.array(W))
    np.save('%s/hid_bias.npy'%save_dir, np.array(hb))
    np.save('%s/vis_bias.npy'%save_dir, np.array(vb))
    np.save('%s/metrics.npy'%save_dir, np.array(metrics))
    
    np.save('%s/trained_weights.npy'%save_dir, np.array(W)[-1])
    np.save('%s/trained_hid_bias.npy'%save_dir, np.array(hb)[-1])
    np.save('%s/trained_vis_bias.npy'%save_dir, np.array(vb)[-1])
    
    return
    
main(parser.parse_args())
