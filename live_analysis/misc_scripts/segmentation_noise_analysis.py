#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 12:21:50 2022

@author: xies
"""

import numpy as np
import pandas as pd
import pickle as pkl

with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/tracked_data_collated/cell_summary.pkl','rb' ) as f:
    df = pkl.load(f)

df = mesa

#%%

noise_level = 100 # % noise leve in fL terms
Nboot = 1000

R_size = np.zeros(Nboot)
R_time = np.zeros(Nboot)
#homeostasis plots
for i in range(Nboot):
    X = df['Birth nuc volume']
    Y = df['Total nuc growth']
    X,Y = nonan_pairs(X,Y)

    X += np.random.normal(0, noise_level, size=X.shape)
    Y += np.random.normal(0, noise_level, size=Y.shape)
    R_size[i] = np.corrcoef(X,Y)[0,1]
    
    
    X = df['Birth nuc volume']
    Y = df['Cycle length']
    X,Y = nonan_pairs(X,Y)

    X += np.random.normal(0, noise_level, size=X.shape)

    R_time[i] = np.corrcoef(X,Y)[0,1]
    
    