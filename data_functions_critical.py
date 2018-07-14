# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:00:56 2018

@author: Stavros
"""

import numpy as np

def add_index(data):
    ## Adds a 1-component dimension to a numpy array to use as input to CNN ##
    return data.reshape(data.shape+(1,))

def read_file(directory, L=16, n_samples=40000):
    ## Returns dataset normilized to [0,1] ##
    data = np.load(directory%(n_samples, L))
    return data.reshape(data.shape[0], L, L)

def temp_partition(data, iT, n_samples=10000):
    return data[iT * n_samples : (iT+1) * n_samples]
    
def data_directory_select(choice_pc, choice_train=0):
    ## Returns default directories for data in SAVVAS-PC and Titan ##
    ## choice_pc = 0 for Titan, 1 for SAVVAS-PC ##
    ## choice_train = 0 for testing, 1 for training ##
    test_or_train = ['test', 'train']
    ending = 'ising-critical-' + test_or_train[choice_train]
    ending += '-%d/L=%d/configs.npy'
    
    if choice_pc == 1:
        ## Fix directory for SAVVAS-PC ##
        starting = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/'
    elif choice_pc == 0:
        ## Fix directory for Titan
        starting = '/home/sefthymiou/super-resolving-ising/ising-data/'
    else:
        starting = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising_Data/'
        
    return starting + ending
