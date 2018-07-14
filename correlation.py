# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 15:00:42 2018

@author: Stavros
"""

import numpy as np

class ComplexExp():
    def __init__(self, phase):
        self.Re = np.cos(phase)
        self.Im = np.sin(phase)
        
class ComplexSigma():
    def __init__(self, state, complex_exp):
        self.Re = np.sum(state * complex_exp.Re, axis=(1,2))
        self.Im = np.sum(state * complex_exp.Im, axis=(1,2))
    
def correlation_lengths_with_errors(state):
    ## Returns S0, S1, S2 and their errors
    
    n_samples, Ly, Lx = state.shape    
    # Create x dot rj
    xrj = np.array(Ly * [np.arange(Lx)])
    
    # Create exponential for q1
    exp_q1 = ComplexExp( - 2*np.pi * xrj / Lx)
    # Create exponential for q2
    exp_q2 = ComplexExp( - 4*np.pi * xrj / Lx)
    
    # Calculate sigmas
    sigma_q0 = np.sum(state, axis=(1,2)) / np.sqrt(Lx * Ly)
    sigma_q = [ComplexSigma(state, exp_q1), ComplexSigma(state, exp_q2)]
    
    # Calculate S
    S = np.zeros(3)
    errS = np.zeros(3) # Add std calculations
    
    sample_sigma = np.square(sigma_q0)
    S[0], errS[0] = np.mean(sample_sigma), np.std(sample_sigma)
    for i in range(2):
        sample_sigma = np.square(sigma_q[i].Re)/ (Lx * Ly)
        + np.square(sigma_q[i].Im)/ (Lx * Ly)
        
        S[i+1], errS[i+1] = np.mean(sample_sigma), np.std(sample_sigma)

    #ksi_a_over_L = np.sqrt(S[0] / S[1] - 1) / (2*np.pi)
    #ksi_b_over_L = np.sqrt((S[1] / S[2] - 1) / (4 - S[1] / S[2])) / (2*np.pi)
    
    #return ksi_a_over_L, ksi_b_over_L, errS
    return S, errS

def correlation_lengths(state):
    ## Returns S0, S1, S2
    
    n_samples, Ly, Lx = state.shape    
    # Create x dot rj
    xrj = np.array(Ly * [np.arange(Lx)])
    
    # Create exponential for q1
    exp_q1 = ComplexExp( - 2*np.pi * xrj / Lx)
    # Create exponential for q2
    exp_q2 = ComplexExp( - 4*np.pi * xrj / Lx)
    
    # Calculate sigmas
    sigma_q0 = np.sum(state, axis=(1,2)) / np.sqrt(Lx * Ly)
    sigma_q = [ComplexSigma(state, exp_q1), ComplexSigma(state, exp_q2)]
    
    # Calculate S
    S = np.zeros(3)
    
    sample_sigma = np.square(sigma_q0)
    S[0] = np.mean(sample_sigma)
    for i in range(2):
        sample_sigma = np.square(sigma_q[i].Re)/ (Lx * Ly)
        + np.square(sigma_q[i].Im)/ (Lx * Ly)
        
        S[i+1] = np.mean(sample_sigma)
        
    #ksi_a_over_L = np.sqrt(S[0] / S[1] - 1) / (2*np.pi)
    #ksi_b_over_L = np.sqrt((S[1] / S[2] - 1) / (4 - S[1] / S[2])) / (2*np.pi)
    
    #return ksi_a_over_L, ksi_b_over_L, errS
    return S

def two_point_function(state, r):
    # Create state duplicate for horizontal correlations
    hor = np.empty(state.shape)
    hor[:, :-r] = state[:, r:]
    hor[:, -r:] = state[:, :r]
    # Calculate the product
    hor *= state
    
    # Repeat for vertical correlations
    ver = np.empty(state.shape)
    ver[:, :, :-r] = state[:, :, r:]
    ver[:, :, -r:] = state[:, :, :r]
    # Calculate the product
    ver *= state
    
    return np.mean(hor + ver) / 2