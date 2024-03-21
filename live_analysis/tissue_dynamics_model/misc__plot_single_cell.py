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

ID = 374

cell2plot = collated[ID]
cell2plot['Num neighbors'] = cell2plot['Num diff neighbors'] + cell2plot['Num planar neighbors']

x = 'Age'
y2plot = ['Volume','Apical area','Basal area'
          ,'Mean neighbor cell volume','Num neighbors','Mean curvature','Height to BM'
          ,'Max neighbor height from BM']
Nsubplots = int(np.ceil(len(y2plot) / 2))

for i,y in enumerate(y2plot):
    
    plt.subplot(2,Nsubplots,i+1)
    
    plt.plot(cell2plot[x],cell2plot[y])
    plt.xlabel(x); plt.ylabel(y)
    
