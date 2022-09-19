#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:36:24 2022

@author: xies
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

from glob import glob
from os import path

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)

#%%

def decollate_df(df,by='basalID'):
    
    indices = df[by]
    all_indices = np.unique(indices)
    
    collated = [df.loc[df[by] == idx] for idx in all_indices]
    return collated

collated = decollate_df(df)

#%% Plot given fields as individual lines

x = 'Time to G1S'
y = 'Basal area'

for i,c in enumerate(collated):
    plt.plot(c[x],c[y])
    plt.xlabel(x)
    plt.ylabel(y)

#%% Plot given fieldas as scatter

x = 'Volume (sm)'
y = 'Specific GR b (sm)'
color = 'Time to G1S'


plt.figure()
sb.scatterplot(data = df,x=x,y=y,hue=color,palette='icefire')

