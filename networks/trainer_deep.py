# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 12:56:37 2018

@author: Stavros
"""

import numpy as np
import tensorflow as tf
from crbm_deep import ConvRBM_Train
#from ising import get_observables_with_corr_and_tpf

class Trainer(ConvRBM_Train):
    def create_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.GPU)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        self.sess.run(tf.global_variables_initializer())
        
    def fit_and_save(self, train_data, val_data, directory):
        self.create_session()
        
        if self.args.ES:
            metrics_mse, metrics_rest = self.fit_early_stopping(train_data, val_data)
            np.save('%s/metrics_mse.npy'%directory, metrics_mse)
            np.save('%s/metrics_rest.npy'%directory, metrics_rest)
            
        else:
            metrics = self.fit(train_data, val_data)
            np.save('%s/metrics.npy'%directory, metrics)
        
        ## Save weights
        np.save('%s/trained_hid_bias.npy'%directory, self.sess.run(self.hid_bias))
        np.save('%s/trained_vis_bias.npy'%directory, self.sess.run(self.vis_bias))
        for (i, f) in enumerate(self.filters):
            np.save('%s/trained_weights%d.npy'%(directory, i+1), self.sess.run(f))
    
    def fit(self, train_data, val_data):
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
        metrics = np.zeros([self.args.EP // self.args.MSG, 4])
        # Metrics: (Train MSE, Val MSE, Train FE, Val FE)
        
        ## Initialize learning rate
        self.learning_rate = self.args.LR
        
        for iE in range(self.args.EP):
            np.random.shuffle(train_data)
            for iB in range(train_batches):
                self.sess.run(self.train_op, 
                         feed_dict={self.visible : train_data[iB * self.args.BS : (iB+1) * self.args.BS],
                                    self.learning_rate_plc : self.learning_rate})
            
            ## Linear LR decay
            if iE >= self.args.LREP:
                self.learning_rate -= self.linear_lr_update
            
            ## Calculate MSE in train and validation dataset and free energy ratio
            if iE % self.args.MSG == 0:
                for iB in range(mse_batch_train):
                    v_recon, free_energy_train[iB] = self.sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_train[iB] = np.mean(np.square(v_recon - 
                             train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                    
                for iB in range(mse_batch_val):
                    v_recon, free_energy_val[iB] = self.sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_val[iB] = np.mean(np.square(v_recon - 
                           val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                
                met_ind = iE // self.args.MSG
                metrics[met_ind] = np.array([mse_train.mean(), mse_val.mean(), 
                       free_energy_train.mean(), free_energy_val.mean()])
                
                print('\n%d / %d epochs done!'%(iE+1, self.args.EP))
                print('Train MSE: %.5E\nValidation MSE: %.5E'%tuple(metrics[met_ind, :2]))
                print('Free energy difference: %.5f\n'%(metrics[met_ind, 2] - metrics[met_ind, 3]))
            
        # Metrics: np.array(epochs // messages, 4)
        # 4 = (Train MSE, Val MSE, Train FE, Val FE)
        return metrics
        
    def fit_early_stopping(self, train_data, val_data):
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
        
        ## Initialize lists to keep track of metrics
        metrics_mse_train = [] # (Train MSE)
        metrics_rest = []  # (Val MSE, Train FE, Val FE)
                
        ## Calculate first train MSE
        for iB in range(mse_batch_train):
            v_recon = self.sess.run(self.v_gibbs_recon_op, 
                               feed_dict={self.visible : train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
            mse_train[iB] = np.mean(np.square(v_recon - 
                     train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
        metrics_mse_train.append(mse_train.mean())
        
        ## Initialize learning rate and ES paramaters
        iE, patience = 0, 0
        train_flag = True
        self.learning_rate = self.args.LR
        
        ## Training loop
        while train_flag:
            np.random.shuffle(train_data)
            for iB in range(train_batches):
                self.sess.run(self.train_op, 
                         feed_dict={self.visible : train_data[iB * self.args.BS : (iB+1) * self.args.BS],
                                    self.learning_rate_plc : self.learning_rate})
            
            ## Linear LR decay (deactivate because for early we don't know epoch number a priori)
#            if iE >= self.args.LREP:
#                self.learning_rate -= self.linear_lr_update
            
            ## Calculate MSE in train and validation dataset and free energy ratio
            if iE % self.args.MSG == 0:
                for iB in range(mse_batch_train):
                    v_recon, free_energy_train[iB] = self.sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_train[iB] = np.mean(np.square(v_recon - 
                             train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                    
                for iB in range(mse_batch_val):
                    v_recon, free_energy_val[iB] = self.sess.run([self.v_gibbs_recon_op, self.free_energy_op], 
                                              feed_dict={self.visible : val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_val[iB] = np.mean(np.square(v_recon - 
                           val_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                
                metrics_mse_train.append(mse_train.mean())
                metrics_rest.append(np.array([mse_val.mean(), 
                                              free_energy_train.mean(), 
                                              free_energy_val.mean()]))
                
                print('\n%d epochs done!'%(iE+1))
                print('Train MSE: %.5E\nValidation MSE: %.5E'%(metrics_mse_train[-1], 
                                                               metrics_rest[-1][0]))
                print('Free energy difference: %.5f\n'%(metrics_rest[-1][1] - metrics_rest[-1][2]))
                
            else:
                for iB in range(mse_batch_train):
                    v_recon = self.sess.run(self.v_gibbs_recon_op, 
                                       feed_dict={self.visible : train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]})
                    mse_train[iB] = np.mean(np.square(v_recon - 
                             train_data[iB * self.args.BSC : (iB+1) * self.args.BSC]))
                    
                metrics_mse_train.append(mse_train.mean())
                                
            ## Early Stopping update
            if metrics_mse_train[-2] - metrics_mse_train[-1] < self.args.ESTHR:
                patience += 1
            if patience > self.args.ESPAT:
                train_flag = False
    
            iE += 1
        
        # mse: (epochs, )
        # rest: (epochs // bsc, 3)
        return np.array(metrics_mse_train), np.array(metrics_rest)