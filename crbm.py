# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:39:14 2018

@author: Stavros
"""

import tensorflow as tf

class ConvRBM():
    def __init__(self, Nv, Nw, K):
        # sizes object has: Nv, Nw, K
        self.Nv, self.Nw, self.K = Nv, Nw, K
        self.Nh = self.Nv - self.Nw + 1
        
        self.visible = tf.placeholder(tf.float32)
        self.hidden  = tf.placeholder(tf.float32)
        
        # Initialization according to Hinton Sec. 8
        self.filter = tf.Variable(
                tf.random_normal(shape=(self.Nw, self.Nw, 1, self.K), 
                                 stddev=0.01), dtype=tf.float32)
        
        self.hid_bias = tf.Variable(tf.zeros(shape=(self.K,)), dtype=tf.float32)
        self.vis_bias = tf.Variable(tf.zeros(shape=(1,)), dtype=tf.float32)
    
    def prob_given_v(self, v):
        x = tf.nn.conv2d(v, self.filter, strides=(1,1,1,1), padding="VALID")
        return tf.sigmoid(tf.add(x, self.hid_bias))
    
    def prob_given_h(self, h):
        batch_size = tf.shape(h)[0]
        x = tf.nn.conv2d_transpose(h, self.filter, strides=(1,1,1,1), padding="VALID",
                                   output_shape=(batch_size, self.Nv, 
                                                 self.Nv, 1))
        return tf.sigmoid(tf.add(x, self.vis_bias))
    
    def create_gibbs_sampler(self, k=2):
        # Returns graph for Gibbs sampling starting from a random v.
        # k: number of block samplings
        v = self.visible
        for i in range(k):
            h = self.sample_tensor(self.prob_given_v(v))
            v = self.sample_tensor(self.prob_given_h(h))
        return v
    
    def sample_visible(self, n_samples, k=1):
        h_samples = tf.Variable(self.sample_tensor(
                tf.constant(0.5, shape=(n_samples, self.Nh, self.Nh, self.K))), 
                                  trainable=False)
        
        v_samples = self.sample_tensor(self.prob_given_h(h_samples))
        for i in range(1, k):
            h_samples = self.sample_tensor(self.prob_given_v(v_samples))
            v_samples = self.sample_tensor(self.prob_given_h(h_samples))
        
        return v_samples
    
    def calculate_energy(self, v, h):
        c = tf.nn.conv2d(v, self.filter, strides=(1,1,1,1), padding="VALID")
        t1 = tf.reduce_sum(tf.multiply(h, c), axis=(1,2,3))
        t2 = tf.reduce_sum(tf.multiply(self.hid_bias, 
                                         tf.reduce_sum(h, axis=(1,2))), axis=1)
        t3 = tf.multiply(self.vis_bias, tf.reduce_sum(v, axis=(1,2,3)))
        
        return -tf.add(t1, tf.add(t2, t3))
    
    def loss_for_grad(self, v, k=2):
        # sample hidden from data
        h_data = tf.stop_gradient(self.sample_tensor(self.prob_given_v(v)))
        data_term = self.calculate_energy(v, h_data)
        
        # CD-k
        h_recon = tf.stop_gradient(self.sample_tensor(self.prob_given_v(v)))
        v_recon = self.sample_tensor(self.prob_given_h(h_recon))
        for i in range(1, k):
            h_recon = self.sample_tensor(self.prob_given_v(v_recon))
            v_recon = self.sample_tensor(self.prob_given_h(h_recon))
        h_recon = self.prob_given_v(v_recon)
        recon_term = self.calculate_energy(v_recon, h_recon)
        
        return tf.reduce_mean(tf.subtract(data_term, recon_term))
    
    def prepare_training(self, k=2, learning_rate=0.01):
        loss = self.loss_for_grad(v=self.visible, k=k)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train = optimizer.minimize(loss)
        
    def free_energy(self, v):
        x = tf.nn.conv2d(v, self.filter, strides=(1,1,1,1), padding="VALID")
        x = tf.add(tf.reduce_sum(x, axis=-1), tf.reduce_sum(self.hid_bias))
        t = tf.multiply(self.vis_bias, tf.reduce_sum(v, axis=(1,2,3)))
        return -tf.add(t, tf.reduce_sum(tf.nn.softplus(x), axis=(1,2)))
    
    def mean_free_energy(self):
        ## Creates a graph for free energy calculation
        return tf.reduce_mean(self.free_energy(self.visible))
        
    @staticmethod
    def sample_tensor(prob):
        # Returns binary tensor according to probability distribution prob
        shape = tf.shape(prob)
        return tf.where(
            tf.less(tf.random_uniform(shape=shape), prob),
            tf.ones(shape=shape),
            tf.zeros(shape=shape))
    
#Nv, Nw, K = 8, 4, 10
#n_samples = 10000
#batch_size = 1000
#epochs = 30
#n_batches = n_samples // batch_size
#
#crbm = ConvRBM(Nv=Nv, Nw=Nw, K=K)
#
#data = np.ones((n_samples, Nv, Nv, 1))
#rbm = ConvRBM(Nv, Nw, K)
#rbm.prepare_training(k=10)
#v_gibbs = rbm.create_gibbs_sampler(k=4)
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#for iE in range(epochs):
#    for iB in range(n_batches):
#        sess.run(rbm.train, 
#                 feed_dict={rbm.visible : data[iB * batch_size : (iB+1) * batch_size]})
#        
#    print('%d / %d epochs done!'%(iE+1, epochs))
#    
## Test
#v_test = sess.run(v_gibbs, feed_dict={rbm.visible : np.random.randint(
#        0, 2, size=(5000, Nv, Nv, 1))})
