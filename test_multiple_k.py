# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:05:48 2018

@author: User
"""

import numpy as np
import data.loaders as dl
from networks.tester import Tester
from argparse import ArgumentParser

parser = ArgumentParser()
## Data parameters
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-iT', type=int, default=0, help='temperature index')
parser.add_argument('-L', type=int, default=16, help='configuration size')

## Model parameters
parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-GPU', type=float, default=0.4, help='GPU fraction')

## Test parameters
parser.add_argument('-nTE', type=int, default=50000, help='RBM sampling test size')
parser.add_argument('-BSC', type=int, default=5000, help='calculation batch size')
parser.add_argument('-kI', type=int, default=5, help='Gibbs sampling k')
parser.add_argument('-kF', type=int, default=40, help='Gibbs sampling k')

def main(args):
    if args.CR:
        critical_n_samples = {8 : 40000, 16 : 100000}
        T = 2 / np.log(1 + np.sqrt(2))
        
        data = dl.read_file_critical(L=args.L, n_samples=critical_n_samples[args.L], train=False)
        
    else:
        from data.directories import T_list
        T = T_list[args.iT]
        
        data = dl.temp_partition(dl.read_file(L=args.L, n_samples=10000, train=False),
                                 args.iT)
    
    
    ## Initialize
    rbm = Tester(args, T=T)
    print('\nInitiating testing with %s.\n'%rbm.name)
    
    ## Test
    rbm.prepare_testing(data)
    obs_list = rbm.test()
    
    ## Save
    save_dir = 'Observables/%s'%['T%.4f'%T, 'Critical'][int(args.CR)]
    dl.create_directory(save_dir)
    save_dir += '/%s'%rbm.name
    dl.create_directory(save_dir)
    np.save('%s/k%d_%d.npy'%(save_dir, args.kI, args.kF), obs_list)
    
    ## Saved .npy format: Array (different k (0 = correct), 12 observables)
    return

main(parser.parse_args())
