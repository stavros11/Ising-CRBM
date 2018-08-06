# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:17:14 2018

@author: Stavros
"""

import numpy as np
import matplotlib.pyplot as plt
T_list = np.linspace(0.01, 4.538, 32)
file_dir = 'C:/Users/Stavros/Documents/Scripts_and_programs/Ising-CRBM-Data'

#from matplotlib import rcParams
#rcParams.update({'font.size': 18})

LOAD_WEIGHTS = True

L = 8
BS = 50
Nw = 8
K = 64
VER = 3
kI, kF = 1, 200

NAME = 'CRBML%d_W%dK%dBS%dLR0.00100_VER%d_ES'%(L, Nw, K, BS, VER)

obs = np.load('%s/Observables/%s/k%d_%d.npy'%(file_dir, NAME, kI, kF))

if LOAD_WEIGHTS:
    w, mse = [], []
    for (iT, T) in enumerate(T_list):
        fdir = '%s/Observables/%s/T%.4f/'%(file_dir, NAME, T)
        w.append(np.load(fdir + 'trained_weights.npy'))
        mse.append(np.load(fdir + 'metrics_mse.npy'))
        
    w = np.array(w)

 
quant_list = ['$M$', '$E$', '$\chi $', '$C_V$', '$M^2$', '$M^4$', '$E^2$',
              '$C(L/2)$', '$C(L/4)$', '$S_0$', '$S_1$', '$S_2$']

def plot_quantity(iT=0, q=0, figsize=(7, 4), kmax=None):
    plt.figure(figsize=figsize)
    if kmax == None:
        plt.plot(np.arange(kI, kF+1), obs[iT, 1:, q], color='blue')
    else:
        plt.plot(np.arange(kI, kmax+1), obs[iT, 1:kmax+1, q], color='blue')
    plt.axhline(y=obs[iT, 0, q], linestyle='--', color='k')
    plt.xlabel('$k$', fontsize=20)
    plt.ylabel(quant_list[q], fontsize=20)
    plt.show()
    
def plot_four(iT=0, figsize=(10, 6), kmax=None):
    plt.figure(figsize=figsize)
    
    for q in range(4):
        plt.subplot(221+q)
        if kmax == None:
            plt.plot(np.arange(kI, kF+1), obs[iT, 1:, q], color='blue')
        else:
            plt.plot(np.arange(kI, kmax+1), obs[iT, 1:kmax+1, q], color='blue')
        plt.axhline(y=obs[iT, 0, q], linestyle='--', color='k')
        plt.xlabel('$k$', fontsize=20)
        plt.ylabel(quant_list[q], fontsize=20)
    plt.show()
    
def plot_four_converged(figsize=(10, 6)):
    plt.figure(figsize=figsize)
    for q in range(4):
        plt.subplot(221+q)
        plt.plot(T_list, obs[:, 0, q], '-o', color='blue')
        plt.plot(T_list, obs[:, -1, q], '-o', color='red')
        plt.axvline(x=2/np.log(1+np.sqrt(2)), linestyle='--', color='black')
        plt.xlabel('$T$')
        plt.ylabel(quant_list[q])
    plt.show()
    
def weight_hist(iT=0, xlim=2, bins=100, figsize=(7, 4)):
    plt.figure(figsize=figsize)
    plt.hist(w[iT].ravel(), bins=bins)
    plt.xlim((-xlim, xlim))
    plt.show()

def plot_mse(iT=0, figsize=(7, 4)):
    #msg_step = len(mse[iT]) // len(met)
    
    plt.figure(figsize=figsize)
    plt.plot(np.arange(0, len(mse[iT])), mse[iT], color='blue')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    plt.show()