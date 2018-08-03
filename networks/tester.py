# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:10:58 2018

@author: Stavros
"""

import numpy as np
import tensorflow as tf
from os import listdir
from crbm import ConvRBM
from ising import get_observables_with_corr_and_tpf

class Tester(ConvRBM):
    def __init__(self, args, T=0.0):
        self.args = args
        self.T = T
        
        save_dir = 'Trained_Models/%s'%['T%.4f'%T, 'Critical'][int(self.args.CR)]
        self.name = listdir(save_dir)[self.args.Mind]
        
        save_dir += '/%s/'%self.name
        
        w = np.load(save_dir + 'trained_weights.npy')
        hb = np.load(save_dir + 'trained_hid_bias.npy')
        vb = np.load(save_dir + 'trained_vis_bias.npy')
        
        self.Nv = self.args.L
        self.Nw, self.K = w.shape[0], w.shape[-1]
        self.Nh = self.Nv - self.Nw + 1
        
        self.create_basic_parameters(w, hb, vb)
        
    def prepare_testing(self, data):
        self.n_batches = self.args.nTE // self.args.BSC
        
        ## Create ops
        self.init_op = self.create_gibbs_sampler_random(self.args.BSC, k=self.args.kI)
        self.block_op = self.create_gibbs_sampler(k=1)
        
        ## Initialize observable list
        self.obs_list = [get_observables_with_corr_and_tpf(data, self.T)]
        self.n_observables = len(self.obs_list[0])
        
        ## Define Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.GPU)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
                
    def predict_init(self):
        obs_prediction = np.zeros([self.n_batches, self.n_observables])
        pred_samples = np.zeros([self.args.nTE, self.Nv, self.Nv, 1])
        for iB in range(self.n_batches):
            pred_batch = self.sess.run(self.init_op)
            
            obs_prediction[iB] = get_observables_with_corr_and_tpf(pred_batch[:,:,:,0], self.T)
            pred_samples[iB * self.args.BSC : (iB+1) * self.args.BSC] = np.copy(pred_batch)
            
        return pred_samples, obs_prediction.mean(axis=0)
    
    def predict_block(self, data_in):
        obs_prediction = np.zeros([self.n_batches, self.n_observables])
        pred_samples = np.zeros([self.args.nTE, self.Nv, self.Nv, 1])
        for iB in range(self.n_batches):
            pred_batch = self.sess.run(self.block_op, feed_dict={
                    self.visible : data_in[iB * self.args.BSC : (iB+1) * self.args.BSC]})
    
            obs_prediction[iB] = get_observables_with_corr_and_tpf(pred_batch[:,:,:,0], self.T)
            pred_samples[iB * self.args.BSC : (iB+1) * self.args.BSC] = np.copy(pred_batch)
            
        return pred_samples, obs_prediction.mean(axis=0)
    
    def test(self):
        ## Initial prediction
        samples, obs_temp = self.predict_init()
        self.obs_list.append(obs_temp)
        
        ## Block predictions
        for k in range(self.args.kI+1, self.args.kF+1):
            samples, obs_temp = self.predict_block(samples)
            self.obs_list.append(obs_temp)
            
            if (k - self.args.kI) % self.args.MSG == 0:
                ## Print observables
                print('\nCorrect vs Predicted Observables with k=%d:'%k)
                for (cor, pred) in zip(self.obs_list[0], self.obs_list[-1]):
                    print('%.6f  -  %.6f'%(cor, pred))
                print('\n')
        
        return np.array(self.obs_list)
        