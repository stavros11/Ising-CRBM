# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:06:07 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
file_dir = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising-CRBM-Data'

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
    
w = np.load('%s/Observables/%s/%s/trained_weights.npy'%(file_dir, temp_str, NAME))
met = np.load('%s/Observables/%s/%s/metrics.npy'%(file_dir, temp_str, NAME))

def weight_hist(xlim=2, bins=100, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.hist(w.ravel(), bins=bins)
    plt.xlim((-xlim, xlim))
    plt.show()

def plot_mse(figsize=(7, 4)):
    msg_step = EP // len(met)
    
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, EP, msg_step), met.T[0], color='blue')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    plt.show()
