# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:52:31 2018

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
parser.add_argument('-GPU', type=float, default=0.3, help='GPU fraction')

## Test parameters
parser.add_argument('-nTE', type=int, default=50000, help='RBM sampling test size')
parser.add_argument('-REC', type=bool, default=False, help='test reconstructions')
parser.add_argument('-BSC', type=int, default=5000, help='calculation batch size (for memory)')
parser.add_argument('-GBk', type=int, default=5, help='Gibbs sampling k')


def main(args):
    if args.CR:
        T = 2 / np.log(1 + np.sqrt(2))
        
        rbm, rbm_name = get_model(args.Mind, Nv=args.L, critical=True)
        data = dl.read_file_critical(L=args.L, n_samples=100000, train=False)
        
    else:
        from data.directories import T_list
        T = T_list[args.iT]
        
        rbm, rbm_name = get_model(args.Mind, Nv=args.L, critical=False, T=T)
        data = dl.read_file(L=args.L, n_samples=10000, train=False)
    
    ## Calculate MC observables
    obs_correct = get_observables_with_corr_and_tpf(data, T)
        
    n_batches = args.nTE // args.BSC
    
    
    ## Create RBM ops
    if args.REC:
        data = dl.add_index(data)[:args.nTE]
        ops = [rbm.create_gibbs_sampler_random(args.BSC, k=args.GBk), 
           rbm.create_gibbs_sampler(k=args.GBk)]
        obs_prediction = np.zeros([n_batches, 2, len(obs_correct)])
    else:
        ops = rbm.create_gibbs_sampler_random(args.BSC, k=args.GBk)
        obs_prediction = np.zeros([n_batches, len(obs_correct)])
    
    print('\nInitiating testing with %s.\n'%rbm_name)
    
    ## Define Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.GPU)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    
    
    if args.REC:
        for iB in range(n_batches):
            sampled, recon = sess.run(ops, 
                                      feed_dict={rbm.visible : data[iB * args.BSC : (iB+1) * args.BSC]})
            obs_prediction[iB] = np.array([get_observables_with_corr_and_tpf(sampled[:,:,:,0], T),
                                           get_observables_with_corr_and_tpf(recon[:,:,:,0], T)])
        
        sampling_error = np.abs(obs_prediction.mean(axis=0)[0] - obs_correct) * 100 / obs_correct
        reconstr_error = np.abs(obs_prediction.mean(axis=0)[1] - obs_correct) * 100 / obs_correct
    
    else:
        for iB in range(n_batches):
            sampled = sess.run(ops)
            obs_prediction[iB] = get_observables_with_corr_and_tpf(sampled[:,:,:,0], T)
        
        sampling_error = np.abs(obs_prediction.mean(axis=0)- obs_correct) * 100 / obs_correct
    
    print('\nGibbs k=%d'%args.GBk)
    print('\nSampling:')
    print(sampling_error)
    if args.REC:
        print('\nReconstruction:')
        print(reconstr_error)
    
    return

main(parser.parse_args())