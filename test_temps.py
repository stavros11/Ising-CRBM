# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 23:15:03 2018

@author: User
"""

import numpy as np
import data.loaders as dl
from data.directories import T_list
from networks.tester import TesterTemps
from argparse import ArgumentParser

parser = ArgumentParser()
## Data parameters
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-iT', type=int, default=0, help='temperature index')
parser.add_argument('-L', type=int, default=16, help='configuration size')

## Model parameters
parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-GPU', type=float, default=0.4, help='GPU fraction')
parser.add_argument('-MSG', type=int, default=10, help='k step for messages')

## Test parameters
parser.add_argument('-nTE', type=int, default=50000, help='RBM sampling test size')
parser.add_argument('-BSC', type=int, default=5000, help='calculation batch size')
parser.add_argument('-kI', type=int, default=1, help='Gibbs sampling k')
parser.add_argument('-kF', type=int, default=200, help='Gibbs sampling k')

def main(args):
    data = dl.read_file(L=args.L, n_samples=10000, train=False)
    
    obs_list = []
    for (iT, T) in enumerate(T_list):
        ## Initialize
        args.iT = iT
        rbm = TesterTemps(args, T=T)
        print('\nInitiating testing with %s.\n'%rbm.name)
        
        ## Test
        rbm.prepare_testing(dl.temp_partition(data, iT))
        obs_list.append(rbm.test())
        
        print('Temperature %d / %d done!'%(iT+1, len(T_list)))
        
    obs_list = np.array(obs_list)
        
    ## Save
    save_dir = 'Observables/%s'%rbm.name
    dl.create_directory(save_dir)
    
    np.save('%s/k%d_%d.npy'%(save_dir, args.kI, args.kF), obs_list)
    ## Saved .npy format: Array (temps, different k (0 = correct), 12 observables)
    
    # Save only the last converged k
    np.save('%s/k%d_converged.npy'%(save_dir, args.kF), 
            np.concatenate((obs_list[:, 0, :], obs_list[:, -1, :])))
    ## Saved .npy format: Array ((0=correct, 1=converged),temps, 12 observables)
     
    return

main(parser.parse_args())