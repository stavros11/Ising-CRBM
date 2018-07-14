# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:00:56 2018

@author: Stavros
"""

import numpy as np
from os import path, mkdir

def read_file(directory, L=16, n_samples=10000, q=2):
    ## Returns dataset normilized to [0,1] ##
    data = np.load(directory%(n_samples, L, q))
    return data.reshape(data.shape[0], L, L)

def add_index(data):
    ## Adds a 1-component dimension to a numpy array to use as input to CNN ##
    return data.reshape(data.shape+(1,))

def data_directory_select(choice_pc, choice_train=0):
    ## Returns default directories for data in SAVVAS-PC and Titan ##
    ## choice_pc = 0 for Titan, 1 for SAVVAS-PC ##
    ## choice_train = 0 for testing, 1 for training ##
    test_or_train = ['test', 'train']
    ending = 'ising-data-' + test_or_train[choice_train]
    ending += '-%d/L=%d/q=%d/configs.npy'
    
    if choice_pc == 1:
        ## Fix directory for SAVVAS-PC ##
        starting = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/'
    elif choice_pc == 0:
        ## Fix directory for Titan
        starting = '/home/sefthymiou/super-resolving-ising/ising-data/'
    else:
        starting = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising_Data/'
        
    return starting + ending

def temp_partition(data, i, samples=10000):
    ## Returns a partition of the dataset corresponding to one temperature ##
    return data[i * samples : (i+1) * samples]

def create_directory(d):
    ## Create directory if it doesn't exist ##
    if not path.exists(d):
        mkdir(d)
