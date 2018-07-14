# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 15:13:40 2018

@author: Stavros
"""

##################################################################
############ Functions that calculate quantities for     #########
############ the 2D Ising model. Uses numpy (not keras)  #########
##################################################################

import numpy as np
from correlation import correlation_lengths, two_point_function

def get_basic_observables(state, T):
    ## Returns [Mag, En, Susc, specHeat]
    obj = Ising(2 * state - 1)
    obj.calculate_moments()
    
    specHeat = (obj.energy2 - np.square(obj.energy)) / T**2
    susc = (obj.mag2 - np.square(obj.mag)) / T
    
    return np.array([obj.mag, obj.energy, susc, specHeat]) / obj.N_spins

def get_observables(state, T):
    ## Returns [Mag, En, Susc, specHeat, Mag2, Mag4, En2]
    return get_observables_assist(2*state-1, T)
    
def get_observables_assist(state, T):
    obj = Ising(state)
    obj.calculate_moments()
    
    specHeat = (obj.energy2 - np.square(obj.energy)) / T**2
    susc = (obj.mag2 - np.square(obj.mag)) / T
    
    return np.array([obj.mag, obj.energy, susc, specHeat,
                     obj.mag2 / obj.N_spins, obj.mag4 / obj.N_spins**3,
                     obj.energy2 / obj.N_spins]) / obj.N_spins

def get_observables_with_corr(state, T):
    ## Returns [Mag, En, Susc, specHeat, Mag2, Mag4, En2, S0, S1, S2]
    state = 2 * state - 1
    obs = get_observables_assist(state, T)
    corr = correlation_lengths(state)
    return np.concatenate((obs, corr))

def get_observables_with_tpf(state, T):
    ## Returns [Mag, En, Susc, specHeat, Mag2, Mag4, En2, tpf(L/2), tpf(L/4)]
    state = 2 * state - 1
    L = state.shape[1]
    
    obs = get_observables_assist(state, T)
    tpf1 = two_point_function(state, L//2)
    tpf2 = two_point_function(state, L//4)
    
    return np.concatenate((obs, np.array([tpf1]), np.array([tpf2])))

def get_observables_with_corr_and_tpf(state, T):
    ## Returns [Mag, En, Susc, specHeat, Mag2, Mag4, En2, 
    ## tpf(L/2), tpf(L/4), S0, S1, S2,]
    state = 2 * state - 1
    L = state.shape[1]
    
    obs = get_observables_assist(state, T)
    corr = correlation_lengths(state)
    tpf1 = two_point_function(state, L//2)
    tpf2 = two_point_function(state, L//4)
    
    return np.concatenate((obs, np.array([tpf1]), np.array([tpf2]), corr))

def get_moments_with_errors(state):
    ## Returns array of [quantity, error] (5x2)
    ## Quantities Mag, Mag2, Mag4, En, En2
    obs = np.empty([5, 2])
    obj = IsingErrors(2 * state - 1)
    obj.calculate_moments()
    
    obs[0] = np.array([obj.mag, obj.errmag])
    obs[1] = np.array([obj.mag2, obj.errmag2])
    obs[2] = np.array([obj.mag4, obj.errmag4])
    obs[3] = np.array([obj.energy, obj.errenergy])
    obs[4] = np.array([obj.energy2, obj.errenergy2])
    
    return obs
    

##########################################
########## CALCULATIONS CLASSES ##########
##########################################

class Ising():
    def __init__(self, state):
        (self.n_samples, self.Ly, self.Lx) = state.shape
        self.N_spins = self.Lx * self.Ly
        self.state = state
        
    def _calculate_magnetization(self):
        self.sample_mag = np.sum(self.state, axis=(1,2))
        
    def _calculate_energy(self, Jx=1, Jy=1):
        ## Returns total energy of the current state - Full calculation ##
        # Energy from x interactions
        Ex = np.sum(self.state[:,:,1:] * self.state[:,:,:self.Lx-1], axis=(1,2))
        # Energy from y interactions
        Ey = np.sum(self.state[:,1:,:] * self.state[:,:self.Ly-1,:], axis=(1,2))
        
        # Fix periodic boundary conditions
        Ex+= np.sum(self.state[:,:,0] * self.state[:,:,self.Lx-1], axis=1)
        Ey+= np.sum(self.state[:,0,:] * self.state[:,self.Ly-1,:], axis=1)
        
        self.sample_energy = - (Jx * Ex + Jy * Ey)
    
    def calculate_moments(self):
        self._calculate_magnetization()
        self.mag  = np.mean(np.abs(self.sample_mag))
        self.mag2 = np.mean(np.square(self.sample_mag))
        self.mag4 = np.mean(self.sample_mag**4)
        
        self._calculate_energy()
        self.energy  = np.mean(self.sample_energy)
        self.energy2 = np.mean(np.square(self.sample_energy))

    
class IsingErrors():
    def __init__(self, state):
        (n_samples, Ly, Lx) = state.shape
        self.N_spins = Lx * Ly
        self.state = state
        
    def average_and_std(self, x):
        return np.mean(x), np.std(x)
    
    def _calculate_magnetization(self):
        ## Fixed to calculate the magnetization of many samples of states simultaneously ##
        ## Returns the average magnetization over all samples ##
        self.sample_mag = np.mean(self.state, axis=(1,2))
        
    def _calculate_energy(self, Jx=1, Jy=1):
        ## Returns total energy of the current state - Full calculation ##
        # Energy from x interactions
        Ex = np.sum(self.state[:,:,1:] * self.state[:,:,:self.Lx-1], axis=(1,2))
        # Energy from y interactions
        Ey = np.sum(self.state[:,1:,:] * self.state[:,:self.Ly-1,:], axis=(1,2))
        
        # Fix periodic boundary conditions
        Ex+= np.sum(self.state[:,:,0] * self.state[:,:,-1], axis=1)
        Ey+= np.sum(self.state[:,0,:] * self.state[:,-1,:], axis=1)
        
        self.sample_energy = - (Jx * Ex + Jy * Ey)
    
    def calculate_moments(self):
        self._calculate_magnetization()
        self.mag, self.errmag = self.average_and_std(np.abs(self.sample_mag))
        self.mag2, self.errmag2 = self.average_and_std(np.square(self.sample_mag))
        self.mag4, self.errmag4 = self.average_and_std(self.sample_mag**4)
        
        self._calculate_energy()
        self.energy, self.errenergy = self.average_and_std(self.sample_energy)
        self.energy2, self.errenergy2 = self.average_and_std(
                np.square(self.sample_energy))
        
        #return np.array([self.mag, self.energy / self.N_spins, 
        #                 self.N_spins * susc, specHeat / self.N_spins])