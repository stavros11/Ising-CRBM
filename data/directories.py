# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:54:49 2018

@author: User
"""

######################################################
############ File directories definitions ############
######################################################

## Temperature list for mc data ##
from numpy import linspace
T_list = linspace(0.01, 4.538, 32)

BASIC_DIR = '/home/sefthymiou/super-resolving-ising/'
#BASIC_DIR = 'C:/Users/Stavros.SAVVAS-PROBOOK/Documents/Scripts_and_Programs/Super_resolution_Titan_scripts/Ising_Data/'

## Data directories ##
mc_train_dir = BASIC_DIR + 'ising-data/ising-data-train-%d/L=%d/q=%d/configs.npy'
mc_test_dir = BASIC_DIR + 'ising-data/ising-data-train-%d/L=%d/q=%d/configs.npy'
mc_critical_train_dir = BASIC_DIR + 'ising-data/ising-critical-train-%d/L=%d/configs.npy'
mc_critical_test_dir = BASIC_DIR + 'ising-data/ising-critical-test-%d/L=%d/configs.npy'

## Network directories ##
models_save_dir = BASIC_DIR + 'Models'
metrics_save_dir = BASIC_DIR + 'Metrics'

models_critical_save_dir = BASIC_DIR + 'ModelsCritical'
metrics_critical_save_dir = BASIC_DIR + 'MetricsCritical'

## Quantities directories ##
quantities_dir = BASIC_DIR + 'Quantities'
quantities_critical_dir = BASIC_DIR + 'QuantitiesCritical'
multiple_exponents_dir = BASIC_DIR + 'MultipleExponents'

## Output directories ##
output_dir = BASIC_DIR + 'Output'
output_critical_dir = BASIC_DIR + 'OutputCritical'