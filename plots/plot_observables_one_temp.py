# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:37:10 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
file_dir = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising-CRBM-Data'

#from matplotlib import rcParams
#rcParams.update({'font.size': 18})

CRIT = True
iT = 20

L = 8
BS = 50
EP = 2000
Nw = 8
K = 64
VER = 1

NAME = 'CRBML%d_W%dK%dBS%dLR0.00100_VER%d_EP%d'%(L, Nw, K, BS, VER, EP)

if CRIT:
    temp_str = 'Critical'
else:
    T_list = np.linspace(0.01, 4.538, 32)
    temp_str = 'T%.4f'%T_list[iT]

## Observables ##
kI, kF = 1, 3000
obs = np.load('%s/Observables/%s/%s/k%d_%d.npy'%(file_dir, temp_str, NAME, kI, kF))

quant_list = ['$M$', '$E$', '$\chi $', '$C_V$', '$M^2$', '$M^4$', '$E^2$',
              '$C(L/2)$', '$C(L/4)$', '$S_0$', '$S_1$', '$S_2$']

errors = np.abs((obs[1:] - obs[0]) / obs[0]) * 100

def plot_quantity(q=0, figsize=(7, 4), kmax=None):
    plt.figure(figsize=figsize)
    if kmax == None:
        plt.plot(np.arange(kI, kF+1), obs[1:, q], color='blue')
    else:
        plt.plot(np.arange(kI, kmax+1), obs[1:kmax+1, q], color='blue')
    plt.axhline(y=obs[0, q], linestyle='--', color='k')
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel(quant_list[q], fontsize=20)
    plt.show()
    
def plot_four(figsize=(10, 6), kmax=None):
    plt.figure(figsize=figsize)
    
    for q in range(4):
        plt.subplot(221+q)
        if kmax == None:
            plt.plot(np.arange(kI, kF+1), obs[1:, q], color='blue')
        else:
            plt.plot(np.arange(kI, kmax+1), obs[1:kmax+1, q], color='blue')
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
