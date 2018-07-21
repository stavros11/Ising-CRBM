# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:15:21 2018

@author: User
"""

import numpy as np
import tensorflow as tf
from crbm import ConvRBM_Train
from ising import get_observables_with_corr_and_tpf

class Trainer(ConvRBM_Train):
    def fit(self, data, T):
        n_batches = data.shape[0] // self.args.BS
        obs_correct = get_observables_with_corr_and_tpf(data[:,:,:,0], T)
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.GPU)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                
        sess.run(tf.global_variables_initializer())
        ## Initialize lists to keep track of weights
        self.W_list, self.hid_bias_list, self.vis_bias_list = [], [], []
        self.update_weight_lists(sess)
        for iE in range(self.args.EP):
            np.random.shuffle(data)
            for iB in range(n_batches):
                sess.run(self.train_op, 
                         feed_dict={self.visible : data[iB * self.args.BS : (iB+1) * self.args.BS]})
            
            self.update_weight_lists(sess)
            ## MSE Test
            if iE % self.args.MSG == 0:
                v_recon = sess.run(self.v_gibbs_op, feed_dict={self.visible : data})
                mse_error = np.mean(np.square(v_recon - data))
                #v_rand = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(0, 2, data.shape)})
                #mse_error_random = np.mean(np.square(v_rand - data))   ## GIVES 0.5 as it should
                print('\n%d / %d epochs done!\nMSE: %.5E\n'%(iE+1, self.args.EP, mse_error))
            
            ## Observables Test
            if iE % self.args.MSGC == 0:
                #v_test = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(
                #        0, 2, size=(args.nTE, args.L, args.L, 1))})[:,:,:,0]
                v_test = sess.run(self.v_gibbs_rand_op)[:,:,:,0]
                obs_test = get_observables_with_corr_and_tpf(v_test, T)
                err_obs = np.abs(obs_test - obs_correct) * 100 / obs_correct
                print(err_obs)
            
            ## Weight averaging update
            if iE >= self.args.WAVEP and (iE - self.args.WAVEP) % self.args.WAVST == 0:
                self.update_network_weights(sess)

    def update_weight_lists(self, sess):
        W_temp, hid_bias_temp, vis_bias_temp = sess.run([self.filter, self.hid_bias, self.vis_bias])
        self.W_list.append(W_temp)
        self.hid_bias_list.append(hid_bias_temp)
        self.vis_bias_list.append(vis_bias_temp)
        
    def update_network_weights(self, sess):
        ## Average weights during training
        W_new = np.array(self.W_list)[-self.args.WAVNR:].mean(axis=0)
        hid_bias_new = np.array(self.hid_bias_list)[-self.args.WAVNR:].mean(axis=0)
        vis_bias_new = np.array(self.vis_bias_list)[-self.args.WAVNR:].mean(axis=0)
        
        sess.run([self.assign_filter, self.assign_hid_bias, self.assign_vis_bias], 
                 feed_dict = {self.filter_plc : W_new, 
                              self.hid_bias_plc : hid_bias_new, 
                              self.vis_bias_plc : vis_bias_new})
