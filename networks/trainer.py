# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:15:21 2018

@author: User
"""

import numpy as np
import tensorflow as tf
from crbm import ConvRBM_Train
#from ising import get_observables_with_corr_and_tpf

class Trainer(ConvRBM_Train):
    def fit(self, train_data, val_data, T):
        train_batches = train_data.shape[0] // self.args.BS
#        obs_correct = get_observables_with_corr_and_tpf(train_data[:,:,:,0], T)
        
        ## Calculation Batches (for memory)
        mse_batch_train = train_data.shape[0] // self.args.BSC
        mse_batch_val = val_data.shape[0] // self.args.BSC
#        obs_batches = self.args.nTE // self.args.BSC
        
        ## Initialize calculation arrays (for batch calculation)
        mse_train, mse_val = np.zeros(mse_batch_train), np.zeros(mse_batch_val)
        free_energy_train, free_energy_val = np.zeros(mse_batch_train), np.zeros(mse_batch_val)
#        obs_test = np.zeros([obs_batches, len(obs_correct)])
        
        ## Initialize lists to keep track of weights and metrics
        self.W_list, self.hid_bias_list, self.vis_bias_list = [], [], []
        metrics = np.zeros([self.args.EP // self.args.MSG, 4])
        # Metrics: (Train MSE, Val MSE, Train FE, Val FE)
        
        ## Define Session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.GPU)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                
        sess.run(tf.global_variables_initializer())
        self.update_weight_lists(sess)
        
        
        ## Initialize learning rate
        self.learning_rate = self.args.LR
        
        for iE in range(self.args.EP):
            np.random.shuffle(train_data)
            for iB in range(train_batches):
                sess.run(self.train_op, 
                         feed_dict={self.visible : train_data[iB * self.args.BS : (iB+1) * self.args.BS],
                                    self.learning_rate_plc : self.learning_rate})
            
            self.update_weight_lists(sess)
            ## Linear LR decay
            if iE >= self.args.LREP:
                self.learning_rate -= self.linear_lr_update
            
            ## Calculate MSE in train and validation dataset and free energy ratio
            if iE % self.args.MSG == 0:
                for iB in range(mse_batch_train):
                    v_recon, free_energy_train[iB] = sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_train[iB] = np.mean(np.square(v_recon - 
                             train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                    
                for iB in range(mse_batch_val):
                    v_recon, free_energy_val[iB] = sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_val[iB] = np.mean(np.square(v_recon - 
                           val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                
                met_ind = iE // self.args.MSG
                metrics[met_ind] = np.array([mse_train.mean(), mse_val.mean(), 
                       free_energy_train.mean(), free_energy_val.mean()])
                
                print('\n%d / %d epochs done!'%(iE+1, self.args.EP))
                print('Train MSE: %.5E\nValidation MSE: %.5E'%tuple(metrics[met_ind, :2]))
                print('Free energy difference: %.5f\n'%(metrics[met_ind, 2] - metrics[met_ind, 3]))
                
#            ## Observables Test
#            if iE % self.args.MSGT == 0:
#                for iB in range(obs_batches):
#                    v_test = sess.run(self.v_gibbs_test_op)[:,:,:,0]
#                    obs_test[iB] = get_observables_with_corr_and_tpf(v_test, T)
#                err_obs = np.abs(obs_test.mean(axis=0) - obs_correct) * 100 / obs_correct
#                print(err_obs)
            
            ## Weight averaging update
            if iE >= self.args.WAVEP and (iE - self.args.WAVEP) % self.args.WAVST == 0:
                self.update_network_weights(sess)
                
        return self.W_list, self.hid_bias_list, self.vis_bias_list, metrics

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
