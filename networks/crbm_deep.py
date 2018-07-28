# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:07:19 2018

@author: User
"""

import tensorflow as tf

class ConvRBM():
    def __init__(self, Nv, filters, hid_b, vis_b):
        # filters: list of saved weights
        # hid_bias: (K,)
        # vis_bias: (1,)
        self.Nv = Nv
        self.visible = tf.placeholder(tf.float32)
        
        self.hid_bias = tf.Variable(hid_b, dtype=tf.float32)
        self.vis_bias = tf.Variable(vis_b, dtype=tf.float32)
        
        self.filters, self.Nw, self.K = [], [], []
        for f in filters:
            self.filters.append(tf.Variable(f, dtype=tf.float32))
            self.Nw.append(f.shape[0])
            self.K.append(f.shape[-1])
        
    def prob_given_v(self, v):
        x = self.apply_convolution(v)
        return tf.sigmoid(tf.add(x, self.hid_bias))
    
    def prob_given_h(self, h):
        x = self.apply_inverse_convolution(h, batch_size=tf.shape(h)[0])
        return tf.sigmoid(tf.add(x, self.vis_bias))
    
    def create_gibbs_sampler(self, k=2):
        # Returns graph for Gibbs sampling starting from a random v.
        # k: number of block samplings
        v = self.visible
        for i in range(k):
            h = self.sample_tensor(self.prob_given_v(v))
            v = self.sample_tensor(self.prob_given_h(h))
        return v
    
    def create_gibbs_sampler_random(self, n_samples, k=1):
        h_samples = tf.Variable(self.sample_tensor(
                tf.constant(0.5, shape=(n_samples, self.Nv, self.Nv, self.K[-1]))), 
                                  trainable=False)
        
        v_samples = self.sample_tensor(self.prob_given_h(h_samples))
        for i in range(1, k):
            h_samples = self.sample_tensor(self.prob_given_v(v_samples))
            v_samples = self.sample_tensor(self.prob_given_h(h_samples))
        
        return v_samples
    
    def calculate_energy(self, v, h):
        c = self.apply_convolution(v)
        t1 = tf.reduce_sum(tf.multiply(h, c), axis=(1,2,3))
        t2 = tf.reduce_sum(tf.multiply(self.hid_bias, 
                                         tf.reduce_sum(h, axis=(1,2))), axis=1)
        t3 = tf.multiply(self.vis_bias, tf.reduce_sum(v, axis=(1,2,3)))
        
        return -tf.add(t1, tf.add(t2, t3))
            
    def free_energy(self, v):
        x = self.apply_convolution(v)
        x = tf.add(tf.reduce_sum(x, axis=-1), tf.reduce_sum(self.hid_bias))
        t = tf.multiply(self.vis_bias, tf.reduce_sum(v, axis=(1,2,3)))
        return -tf.add(t, tf.reduce_sum(tf.nn.softplus(x), axis=(1,2)))
                
    def apply_convolution(self, v):
        #return tf.nn.conv2d(v, self.filter, strides=(1,1,1,1), padding="VALID")
        x = self.PBC_padding(v, pad=self.Nw[0]-1)
        x = tf.nn.conv2d(x, self.filters[0], strides=(1,1,1,1), padding="VALID")
        for (n, f) in zip(self.Nw[1:], self.filters[1:]):
            x = self.PBC_padding(x, pad=n-1)
            x = tf.nn.conv2d(x, f, strides=(1,1,1,1), padding="VALID")
        
        return x
        
    def apply_inverse_convolution(self, h, batch_size):
        #return tf.nn.conv2d_transpose(h, self.filter, strides=(1,1,1,1), padding="VALID",
        #                              output_shape=(batch_size, self.Nv, self.Nv, 1))
        x = tf.nn.conv2d_transpose(h, self.filters[-1], strides=(1,1,1,1), padding="VALID",
                                      output_shape=(batch_size, self.Nv+self.Nw[-1]-1, self.Nv+self.Nw[-1]-1, 
                                                    self.K[-2]))
        x = x[:, :self.Nv, :self.Nv]
        for (n, f, k) in zip(self.Nw[::-1][1:], self.filters[::-1][1:], self.K[::-1][2:]+[1]):
            x = tf.nn.conv2d_transpose(x, f, strides=(1,1,1,1), padding="VALID",
                                      output_shape=(batch_size, self.Nv+n-1, self.Nv+n-1, k))
            x = x[:, :self.Nv, :self.Nv]
            
        return x
    
    @staticmethod
    def PBC_padding(x, pad=1):
        L = tf.shape(x)[1]
        y = tf.tile(x, [1, 2, 2, 1])
        
        return y[:, :L+pad, :L+pad]
    
    @staticmethod
    def sample_tensor(prob):
        # Returns binary tensor according to probability distribution prob
        shape = tf.shape(prob)
        return tf.where(
            tf.less(tf.random_uniform(shape=shape), prob),
            tf.ones(shape=shape),
            tf.zeros(shape=shape))
        
class ConvRBM_Train(ConvRBM):
    def __init__(self, args):
        self.args = args
        
        self.Nv = args.L
        self.K = args.K # K = nr of channels 
        self.Nw = args.Nw # filter sizes
               
        self.visible = tf.placeholder(tf.float32, shape=(None, self.Nv, self.Nv, 1))
        self.hidden = tf.placeholder(tf.float32, shape=(None, self.Nv, self.Nv, self.K[-1]))
        
        # Initialization according to Hinton Sec. 8
        self.filters = [tf.Variable(tf.random_normal(shape=(n, n, k1, k2), 
                                 stddev=0.01), dtype=tf.float32
                        ) for (n, k1, k2) in zip(self.Nw, [1]+self.K[:-1], self.K)]
        
        self.hid_bias = tf.Variable(tf.zeros(shape=(self.K[-1],)), dtype=tf.float32)
        self.vis_bias = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)
        
        self.create_name()
        
    def create_name(self):
        self.name = 'CRBMDeepL%d_'%(self.Nv)
        
        wstr = 'W'
        kstr = 'K'
        for (n, k) in zip(self.Nw, self.K):
            wstr += '%d_'%n
            kstr += '%d_'%k
        self.name += wstr + kstr
        
        self.name += 'BS%dLR%.5f_VER%d'%(self.args.BS, self.args.LR, self.args.VER)
        
        if self.args.ES:
            self.name += '_ES'
        else:
            self.name += '_EP%d'%(self.args.EP)
        
    def loss_for_grad(self, v, k=2):
        # sample hidden from data
        h_data = tf.stop_gradient(self.sample_tensor(self.prob_given_v(v)))
        data_term = self.calculate_energy(v, h_data)
        
        # CD-k
#        h_recon = tf.stop_gradient(self.sample_tensor(self.prob_given_v(v)))
#        v_recon = self.sample_tensor(self.prob_given_h(h_recon))
#        for i in range(1, k):
#            h_recon = self.sample_tensor(self.prob_given_v(v_recon))
#            v_recon = self.sample_tensor(self.prob_given_h(h_recon))
#        h_recon = self.prob_given_v(v_recon)
#        recon_term = self.calculate_energy(v_recon, h_recon)
        
        h_samples = self.hidden_samples
        v_samples = tf.stop_gradient(self.sample_tensor(self.prob_given_h(h_samples)))
        h_samples = self.sample_tensor(self.prob_given_v(v_samples))
        for i in range(1, k):
            v_samples = self.sample_tensor(self.prob_given_h(h_samples))
            h_samples = self.sample_tensor(self.prob_given_v(v_samples))
        self.hidden_samples = self.hidden_samples.assign(h_samples)
        
        recon_term = self.calculate_energy(v_samples, self.hidden_samples)
        
        return tf.reduce_mean(tf.subtract(data_term, recon_term))
    
    def prepare_training(self):
        #variables for sampling (num_samples is the number of samples to return):
        # from Torlai PSI tutorial
        self.hidden_samples = tf.Variable(
                self.sample_tensor(tf.constant(0.5, shape=(self.args.BS, self.Nv, self.Nv, self.K[-1]))), 
                                  trainable=False)
        
        ## For learning rate decay
        self.learning_rate_plc = tf.placeholder(tf.float32, shape=[])
        
        if self.args.LREP >= self.args.LR:
            self.linear_lr_update = 0
        else:
            self.linear_lr_update = (self.args.LR - self.args.LRF) / (self.args.EP - self.args.LREP)

        loss = self.loss_for_grad(v=self.visible, k=self.args.GBTR)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_plc)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)
        
        ## Create validation ops
        self.v_gibbs_recon_op = self.create_gibbs_sampler(k=self.args.GBMSE)
        self.v_gibbs_test_op = self.create_gibbs_sampler_random(self.args.BSC, k=self.args.GBTE)
        self.free_energy_op = tf.reduce_mean(self.free_energy(self.visible))
