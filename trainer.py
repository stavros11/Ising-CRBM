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
    def fit(self, data):
        n_batches = data.shape[0] // self.args.BS
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.args.GPU)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.sess.run(tf.global_variables_initializer())
        for iE in range(self.args.EP):
            np.random.shuffle(data)
            for iB in range(n_batches):
                sess.run(self.train_op, 
                         feed_dict={self.visible : data[iB * self.args.BS : (iB+1) * self.args.BS]})
            
            if iE % self.args.MSG == 0:
                v_recon = sess.run(self.v_gibbs_op, feed_dict={self.visible : data})
                mse_error = np.mean(np.square(v_recon - data))
                #v_rand = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(0, 2, data.shape)})
                #mse_error_random = np.mean(np.square(v_rand - data))   ## GIVES 0.5 as it should
                print('\n%d / %d epochs done!\nMSE: %.5E\n'%(iE+1, self.args.EP, mse_error))
            
            ## Test
#            if iE % self.args.MSGC == 0:
#                #v_test = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(
#                #        0, 2, size=(args.nTE, args.L, args.L, 1))})[:,:,:,0]
#                v_test = sess.run(self.v_gibbs_rand)[:,:,:,0]
#                obs_test = get_observables_with_corr_and_tpf(v_test, T)
#                err_obs = np.abs(obs_test - obs_correct) * 100 / obs_correct
#                print(err_obs)