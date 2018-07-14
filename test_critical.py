# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 12:39:25 2018

@author: Stavros
"""

import numpy as np
import tensorflow as tf
import data_functions_critical as df
from crbm import ConvRBM
from ising import get_observables_with_corr_and_tpf
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-EP', type=int, default=500, help='number of epochs')
parser.add_argument('-BS', type=int, default=1000, help='batch size')
parser.add_argument('-L', type=int, default=16, help='configuration size')
parser.add_argument('-nTR', type=int, default=100000, help='training samples')
parser.add_argument('-nTE', type=int, default=50000, help='test samples')
parser.add_argument('-GPU', type=float, default=0.3, help='GPU fraction')
parser.add_argument('-LR', type=float, default=0.01, help='learning rate')

parser.add_argument('-Nw', type=int, default=5, help='filter size')
parser.add_argument('-K', type=int, default=20, help='hidden groups')
parser.add_argument('-GBTR', type=int, default=20, help='Gibbs for CD')
parser.add_argument('-GBTE', type=int, default=1, help='Gibbs for testing')

parser.add_argument('-MSG', type=int, default=10, help='epoch message')
parser.add_argument('-MSGC', type=int, default=40, help='calculation message')

args = parser.parse_args()

T_list = np.linspace(0.01, 4.538, 32)
Tc = 2 / np.log(1 + np.sqrt(2))

## Load data and calculate original observables
data = df.add_index(df.read_file(df.data_directory_select(0), 
                                 L=args.L, n_samples=args.nTR))
n_batches = data.shape[0] // args.BS
obs_correct = get_observables_with_corr_and_tpf(data[:,:,:,0], Tc)

## Define RBM model
rbm = ConvRBM(args.L, args.Nw, args.K)
rbm.prepare_training(k=args.GBTR, learning_rate=args.LR)
v_gibbs = rbm.create_gibbs_sampler(k=args.GBTE)
free_energy = rbm.mean_free_energy()

print('RBM with %d visible units and %d hidden units.'%(rbm.Nv**2, rbm.Nh**2*rbm.K))

## Define tf session with GPU options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.GPU)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

## Train
sess.run(tf.global_variables_initializer())
for iE in range(args.EP):
    np.random.shuffle(data)
    for iB in range(n_batches):
        sess.run(rbm.train, 
                 feed_dict={rbm.visible : data[iB * args.BS : (iB+1) * args.BS]})
    
    if iE % args.MSG == 0:
        en_calc = sess.run(free_energy, feed_dict={rbm.visible : data[:10000]})
        print('%d / %d epochs done! - Free energy: %.5f'%(iE+1, args.EP, en_calc))
    
    ## Test
    if iE % args.MSGC == 0:
        v_test = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(
                0, 2, size=(args.nTE, args.L, args.L, 1))})[:,:,:,0]
    
        obs_test = get_observables_with_corr_and_tpf(v_test, Tc)
        err_obs = np.abs(obs_test - obs_correct) * 100 / obs_correct
        print(err_obs)