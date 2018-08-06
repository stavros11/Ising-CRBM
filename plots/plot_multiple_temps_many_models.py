# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:10:27 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
T_list = np.linspace(0.01, 4.538, 32)
file_dir = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising-CRBM-Data'

#from matplotlib import rcParams
#rcParams.update({'font.size': 18})

K_list = [16, 64, 96]

L = 8
BS = 50
Nw = 8
VER = 3
kI, kF = 1, 200

## Load ##
obs = []
for K in K_list:
    NAME = 'CRBML%d_W%dK%dBS%dLR0.00100_VER%d_ES'%(L, Nw, K, BS, VER)
    obsK = np.load('%s/Observables/%s/k%d_%d.npy'%(file_dir, NAME, kI, kF))
    obs.append(obsK[:, -1, :])
obs_correct = obsK[:, 0, :]
obs = np.array(obs)

quant_list = ['$M$', '$E$', '$\chi $', '$C_V$', '$M^2$', '$M^4$', '$E^2$',
              '$C(L/2)$', '$C(L/4)$', '$S_0$', '$S_1$', '$S_2$']

## Correct chi and Cv
obs[:, 0, 2:4] = 0

def plot_four(figsize=(10, 6), tlims=(1, 3.54)):
    color_list = ['red', 'green', 'blue']
    
    plt.figure(figsize=figsize)
    for q in range(4):
        plt.subplot(221+q)
        # plot correct
        plt.plot(T_list, obs_correct[:, q], '-*', color='black')
        # plot predictions
        for (i, K) in enumerate(K_list):
            plt.plot(T_list, obs[i, :, q], '-*', color=color_list[i], label='$n_H=%d$'%K)
        
        plt.xlim(tlims)
        plt.axvline(x=2/np.log(1+np.sqrt(2)), linestyle='--', color='black')
        plt.xlabel('$T$')
        plt.ylabel(quant_list[q])
        if q == 0:
            plt.legend()
    plt.show()