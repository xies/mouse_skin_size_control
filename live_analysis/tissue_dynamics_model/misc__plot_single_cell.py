#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:35:26 2022

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
    
    collated = {idx: df.loc[df[by] == idx] for idx in all_indices}
    return collated

collated = decollate_df(df)

#%%

ID = 341

x = 'Frame'
y = 'Coronal density'
plt.figure(); plt.plot(collated[ID][x], collated[ID][y])
plt.xlabel(x); plt.ylabel(y)

x = 'Frame'
y = 'Planar angle'
plt.figure(); plt.plot(collated[ID][x], collated[ID][y])
plt.xlabel(x); plt.ylabel(y)

x = 'Frame'
y = 'Collagen orientation'
plt.figure(); plt.plot(collated[ID][x], collated[ID][y])
plt.xlabel(x); plt.ylabel(y)

x = 'Frame'
y = 'Num diff neighbors'
plt.figure(); plt.plot(collated[ID][x], collated[ID][y])
plt.xlabel(x); plt.ylabel(y)

x = 'Frame'
y = 'Basal area'
plt.figure(); plt.plot(collated[ID][x], collated[ID][y])
plt.xlabel(x); plt.ylabel(y)

#%%