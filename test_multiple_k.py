# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 19:05:48 2018

@author: User
"""

import numpy as np
import tensorflow as tf
import data.loaders as dl
from networks.loader import get_model
from networks.ising import get_observables_with_corr_and_tpf
from argparse import ArgumentParser

parser = ArgumentParser()
## Data parameters
parser.add_argument('-CR', type=bool, default=False, help='critical data')
parser.add_argument('-iT', type=int, default=0, help='temperature index')
parser.add_argument('-L', type=int, default=16, help='configuration size')

## Model parameters
parser.add_argument('-Mind', type=int, default=0, help='model index')
parser.add_argument('-GPU', type=float, default=0.5, help='GPU fraction')

## Test parameters
parser.add_argument('-nTE', type=int, default=50000, help='RBM sampling test size')
parser.add_argument('-kI', type=int, default=5, help='Gibbs sampling k')
parser.add_argument('-kF', type=int, default=40, help='Gibbs sampling k')

def main(args):
    if args.CR:
        T = 2 / np.log(1 + np.sqrt(2))
        
        rbm, rbm_name = get_model(args.Mind, Nv=args.L, critical=True)
        data = dl.read_file_critical(L=args.L, n_samples=100000, train=False)
        
    else:
        from data.directories import T_list
        T = T_list[args.iT]
        
        rbm, rbm_name = get_model(args.Mind, Nv=args.L, critical=False, T=T)
        data = dl.temp_partition(dl.read_file(L=args.L, n_samples=10000, train=False),
                                 args.iT)
    
    
    ## Initialize observable list
    obs_list = [get_observables_with_corr_and_tpf(data, T)]
    print('\nInitiating testing with %s.\n'%rbm_name)
    
    for k in range(args.kI, args.kF+1):
        if k < 10:
            BSC = 5000
        elif k < 20:
            BSC = 2000
        else:
            BSC = 1000
        
        ## Create ops
        n_batches = args.nTE // BSC
        obs_prediction = np.zeros([n_batches, len(obs_list[0])])
        ops = rbm.create_gibbs_sampler_random(BSC, k=k)
        
        ## Define Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.GPU)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        
        for iB in range(n_batches):
            sampled = sess.run(ops)
            obs_prediction[iB] = get_observables_with_corr_and_tpf(sampled[:,:,:,0], T)
        
        obs_list.append(obs_prediction.mean(axis=0))
    
        ## Print observables
        print('\nCorrect vs Predicted Observables with k=%d:'%k)
        for (cor, pred) in zip(obs_list[0], obs_list[-1]):
            print('%.6f  -  %.6f'%(cor, pred))
        print('\n')
        
    save_dir = 'Observables/%s'%['T%.4f'%T, 'Critical'][int(args.CR)]
    dl.create_directory(save_dir)
    save_dir += '/%s'%rbm_name
    dl.create_directory(save_dir)
    np.save('%s/k%d_%d.npy'%(save_dir, args.kI, args.kF), np.array(obs_list))
    
    ## Saved .npy format: Array (different k (0 = correct), 12 observables)
    return

main(parser.parse_args())
