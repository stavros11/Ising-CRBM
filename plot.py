# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:37:10 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import rcParams
#rcParams.update({'font.size': 18})

## Metrics ##
#met = np.load('Trained_Models/Critical/CRBML8_W8K64BS50LR0.00100_VER2_EP500/metrics.npy')
#
#plt.plot(np.arange(len(met)), met.T[0])
#plt.show()

## Observables ##
kI, kF = 5, 3000
obs = np.load('Observables/Critical/CRBML8_W8K64BS50LR0.00100_VER1_EP500/k%d_%d.npy'%(kI, kF))

quant_list = ['$M$', '$E$', '$\chi $', '$C_V$', '$M^2$', '$M^4$', '$E^2$',
              '$C(L/2)$', '$C(L/4)$', '$S_0$', '$S_1$', '$S_2$']

errors = np.abs((obs[1:] - obs[0]) / obs[0]) * 100

def plot_quantity(q=0, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.plot(np.arange(kI, kF+1), obs[1:, q], color='blue')
    plt.axhline(y=obs[0, q], linestyle='--', color='k')
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel(quant_list[q], fontsize=20)
    plt.show()
    
def plot_errors(q=0, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.plot(np.arange(kI, kF+1), errors.T[q], color='blue')
    plt.axhline(y=0, linestyle='--', color='k')
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel(quant_list[q] + ' - Error', fontsize=20)
    plt.show()