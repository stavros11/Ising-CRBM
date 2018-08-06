# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:57:02 2018

@author: User
"""

import numpy as np
from os import path, mkdir
from directories import mc_train_dir, mc_test_dir
from directories import mc_critical_train_dir, mc_critical_test_dir
       
###################################
########## LOAD MC DATA  ##########
###################################

def read_file(L=16, q=2, n_samples=10000, train=False):
    ## Returns dataset normilized to [0,1] ##
    if train:
        data = np.load(mc_train_dir%(n_samples, L, q))
    else:
        data = np.load(mc_test_dir%(n_samples, L, q))
    return data.reshape(data.shape[0], L, L)

def read_file_critical(L=16, n_samples=40000, train=False):
    if train:
        return np.load(mc_critical_train_dir%(n_samples, L))
    else:
        return np.load(mc_critical_test_dir%(n_samples, L))
    
####################################
########## DATA UTILITIES ##########
####################################

def add_index(data):
    ## Adds a 1-component dimension to a numpy array to use as input to CNN ##
    return data.reshape(data.shape+(1,))

def temp_partition(data, iT, n_samples=10000):
    return data[iT * n_samples : (iT+1) * n_samples]

def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)
        
###################################
########## PREPROCESSING ##########
###################################

def Z2_zero_majority(data):
    ## Flips spins (preserving Z2 symmetry) so that the majority of them are 0
    states = 2 * data - 1
    flipper = 2 * (states.sum(axis=(1,2)) < 0).astype(np.int) - 1
    
    return (np.multiply(flipper[:, np.newaxis, np.newaxis], states) > 0).astype(np.int)
    
    
    
    