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

with open(path.join(dirname,'basal_with_daughters.pkl'),'rb') as f:
    collated = pkl.load(f)
    
cell_ts = pd.concat(collated,ignore_index=True)

tissue = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col = 0)
# tissue_ = tissue[~np.isnan(tissue['basalID'].values)]

df = pd.merge(cell_ts, tissue, how='inner', on=['basalID','Frame'])
df['Relative nuclear height'] = df['Z_y'] - df['Z_x']

#% Derive cell->tissue features
df['Cell alignment'] = np.abs(np.cos(df['Coronal angle'] - df['Planar angle']))
df['Coronal area'] = df['Coronal area'] - df['Middle area']
df['Coronal density'] = df['Num planar neighbors']/df['Coronal area']

# Derive some neighborhood dynamics
col_idx = len(df.columns)
df['Neighbor mean height frame-1'] = np.nan
df['Neighbor mean height frame-2'] = np.nan
df['FUCCI bg sub frame-1'] = np.nan
df['FUCCI bg sub frame-2'] = np.nan
df['Neighbor mean nuclear volume frame-1'] = np.nan
df['Neighbor mean nuclear volume frame-2'] = np.nan
df['Coronal density frame-1'] = np.nan
df['Coronal density frame-2'] = np.nan
df['Delta curvature'] = np.nan
df['Delta height'] = np.nan
# Call fate based on cell height
# For each cell, load all frames, then grab the prev frame height data
# Some dynamics
for basalID in collated.keys():
    idx = np.where(df['basalID'] == basalID)[0]
    this_len = len(idx)
    if this_len > 1:
        
        this_cell = df.iloc[idx]        
        # Dynamics
        heights = this_cell['Mean neighbor height'].values
        fucci_int = this_cell['FUCCI bg sub'].values
        neighbor_vol = this_cell['Mean neighbor nuclear volume normalized'].values
        cor_density = this_cell['Coronal density'].values
        curvature = this_cell['Mean curvature'].values
        bm_height = this_cell['Height to BM'].values
        
        # Compute d/dt
        # df.at[idx[1:],'Delta curvature'] = np.diff(this_cell['Mean curvature'])
        # df.at[idx[1:],'Delta height'] = np.diff(this_cell['Height to BM'])
        
        # The prev frame
        for t in np.arange(1,this_len):
            # 12h before
            df.at[idx[t],'FUCCI bg sub frame-1'] = fucci_int[t-1]
            df.at[idx[t],'Neighbor mean height frame-1'] = heights[t-1]
            df.at[idx[t],'Neighbor mean nuclear volume frame-1'] = neighbor_vol[t-1]
            df.at[idx[t],'Coronal density frame-1'] = cor_density[t-1]
            df.at[idx[t],'Delta curvature'] = curvature[t] - curvature[t-1]
            df.at[idx[t],'Delta height'] = bm_height[t] - bm_height[t-1]
            
            if t > 1:
                # 24h before
                df.at[idx[t],'Neighbor mean height frame-2'] = heights[t-2]
                df.at[idx[t],'FUCCI bg sub frame-2'] = fucci_int[t-2]
                df.at[idx[t],'Neighbor mean nuclear volume frame-2'] = neighbor_vol[t-2]
                df.at[idx[t],'Coronal density frame-2'] = cor_density[t-2]
        
            
df['NC ratio'] = df['Nuclear volume']/df['Volume (sm)']
df['NC ratio raw'] = df['Nuclear volume raw']/df['Volume (sm)']
df['NC ratio normalized'] = df['Nuclear volume normalized']/df['Volume (sm)']
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

