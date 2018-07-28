# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 18:45:04 2018

@author: User
"""

import numpy as np
from os import listdir
from crbm import ConvRBM

def get_model(model_index, Nv, critical=False, T=0.0):
    save_dir = 'Trained_Models/%s'%['T%.4f'%T, 'Critical'][int(critical)]
    name = listdir(save_dir)[model_index]
    
    save_dir += '/%s/'%name
    
    w = np.load(save_dir + 'trained_weights.npy')
    hb = np.load(save_dir + 'trained_hid_bias.npy')
    vb = np.load(save_dir + 'trained_vis_bias.npy')
    
    return ConvRBM(Nv, filter_w=w, hid_b=hb, vis_b=vb), name
