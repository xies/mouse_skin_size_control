#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 19:58:28 2022

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
ZZ = 72
XX = 460
T = 15

#%% Load

with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
    collated = pkl.load(f)
    
cell_ts = pd.concat(collated,ignore_index=True)

tissue = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col = 0)
# tissue_ = tissue[~np.isnan(tissue['basalID'].values)]

df = pd.merge(cell_ts, tissue, how='inner', on=['basalID','Frame'])
df['Relative nuclear height'] = df['Z_y'] - df['Z_x']

#% Derive cell->tissue features

# @todo: Alignment of cell to local tissue
df['Cell alignment'] = np.abs(np.cos(df['Coronal angle'] - df['Planar angle']))

df['Coronal area'] = df['Coronal area'] - df['Middle area']
df['Coronal density'] = df['Num planar neighbors']/df['Coronal area']

# @todo: look back in time and look at height!

col_idx = len(df.columns)
df['Neighbor mean height frame-1'] = np.nan
df['Neighbor mean height frame-2'] = np.nan
# Call fate based on cell height
# For each cell, load all frames, then grab the prev frame height data
for basalID in collated.keys():
    idx = np.where(df['basalID'] == basalID)[0]
    this_len = len(idx)
    if this_len > 1:
        
        this_cell = df.iloc[idx]
        heights = this_cell['Mean neighbor height'].values
        
        for t in np.arange(1,this_len):
            df.at[idx[t],'Neighbor mean height frame-1'] = heights[t-1]
            if t > 1:
                df.at[idx[t],'Neighbor mean height frame-2'] = heights[t-2]
            
# df['Neighbor max
# df['Neighbor mean height frame -1 or -2'] = np.array([df['Neighbor mean height frame-2'],df['Neighbor mean height frame-1']]).max(axis=0).shape

df.to_csv(path.join(dirname,'MLR model/ts_features.csv'))

df_ = df[df['Phase'] != '?']

#%%

# sb.pairplot(df_,vars=['Volume','Height to BM',
#                      'Mean curvature','Mean neighbor dist','Growth rate (sm)','Specific GR (sm)'
#                      ,'Axial angle','Coronal eccentricity'],plot_kws={'alpha':0.5}
#             , hue='Phase')

# sb.pairplot(df_,vars=['Volume','Collagen fibrousness','Collagen alignment',
#                       'Neighbor mean height frame-2','Neighbor mean height frame-1',
#                       'Specific GR b (sm)','Coronal density'],
#             plot_kws={'alpha':0.5}
#             ,kind='hist')


sb.pairplot(df_,vars=['Volume','Planar component 1','Coronal area',
                      'Neighbor mean height frame-2','Mean neighbor nuclear volume',
                      'Specific GR b (sm)','Phase'],
            plot_kws={'alpha':0.5}
            ,kind='hist')

#%%

sb.regplot(data =df_,y='Phase', logistic=True, x='Growth rate b')

