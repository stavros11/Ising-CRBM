# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:06:07 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt

CRIT = False
iT = 20

L = 8
BS = 50
EP = 5000
Nw = 8
K = 64
VER = 2

NAME = 'CRBML%d_W%dK%dBS%dLR0.00100_VER%d_EP%d'%(L, Nw, K, BS, VER, EP)

if CRIT:
    temp_str = 'Critical'
else:
    from data.directories import T_list
    temp_str = 'T%.4f'%T_list[iT]
    
w = np.load('Observables/%s/%s/trained_weights.npy'%(temp_str, NAME))

def weight_hist(bins=20, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.hist(w.ravel(), bins=bins)
    plt.xlim((-1, 1))
    plt.show()
