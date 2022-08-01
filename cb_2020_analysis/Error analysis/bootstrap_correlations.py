#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:38:11 2022

@author: xies
"""


import numpy as np
import pandas as pd

# with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/tracked_data_collated/cell_summary.pkl','rb' ) as f:
#     df = pkl.load(f)


#%% Convert 12h sampling rate to 24h sampling rate

Nboot = 1000
subsample_size = 40

regression_total = np.zeros(Nboot)
regression_g1 = np.zeros(Nboot)

corr_total = np.zeros(Nboot)
corr_g1 = np.zeros(Nboot)

length_total = np.zeros(Nboot)
length_g1 = np.zeros(Nboot)

for i in range(Nboot):
    
    I = sample(range(len(df)),subsample_size)
    df_ = df.iloc[I]

    regression_total[i] = np.polyfit(*nonan_pairs(df_['Birth nuc volume'],df_['Total nuc growth']),1)[0]
    regression_g1[i] = np.polyfit(*nonan_pairs(df_['Birth nuc volume'],df_['G1 nuc grown']),1)[0]
    
    corr_total[i] = np.corrcoef(*nonan_pairs(df_['Birth nuc volume'],df_['Total nuc growth']))[0,1]
    corr_g1[i] = np.corrcoef(*nonan_pairs(df_['Birth nuc volume'],df_['G1 nuc grown']))[0,1]
    
    length_total[i] = np.corrcoef(*nonan_pairs(df_['Birth nuc volume'],df_['Cycle length']))[0,1]
    length_g1[i] = np.corrcoef(*nonan_pairs(df_['Birth nuc volume'],df_['G1 length']))[0,1]
    
#%%
    
plt.figure()
plt.hist(corr_g1,histtype='step')
plt.hist(corr_total,histtype='step')
plt.xlabel('Corr coef')
plt.legend(['G1','Total'])

plt.figure()
plt.hist(regression_g1,histtype='step')
plt.hist(regression_total,histtype='step')
plt.xlabel('Regression')
plt.legend(['G1','Total'])

plt.figure()
plt.hist(length_total,histtype='step')
plt.hist(length_g1,histtype='step')
plt.xlabel('Time corr')
plt.legend(['G1','Total'])


