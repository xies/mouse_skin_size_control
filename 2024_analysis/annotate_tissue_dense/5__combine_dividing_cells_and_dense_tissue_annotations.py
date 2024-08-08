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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
# dirname = '/Users/xies/Desktop/Code/mouse_skin_size_control/2024_analysis/test_dataset/'

print(f'--- Working on {dirname} ---')

ZZ = 72
XX = 460
T = 15

#%% Cross-reference the same central dividing cell

with open(path.join(dirname,'basal_no_daughters.pkl'),'rb') as f:
    collated = pkl.load(f)
    
cell_ts = pd.concat(collated,ignore_index=True)

tissue = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col = 0)
tissue_cols = set(tissue.columns)
cell_cols = set(cell_ts.columns)
cols2dropintissue = tissue_cols.intersection(cell_cols) - {'basalID','Frame','Z'}
tissue = tissue.drop(columns=cols2dropintissue)

df = pd.merge(cell_ts, tissue, how='inner', on=['basalID','Frame'])
df['Region'] = dirname

# Trim redundant features - will by default use the hand-segmented data
df['Relative nuclear height'] = df['Z_y'] - df['Z_x']

#% Derive cell->tissue features
df['Cell alignment to corona'] = np.abs(np.cos(df['Coronal angle'] - df['Planar angle']))
df['Coronal area'] = df['Coronal area'] - df['Middle area']
df['Coronal density'] = df['Num planar neighbors']/df['Coronal area']

# Derive cell-intrinsic dynamics
df['FUCCI bg sub frame-1'] = np.nan
df['FUCCI bg sub frame-2'] = np.nan
df['Delta curvature'] = np.nan
df['Delta height'] = np.nan
df['Volume frame-1'] = np.nan
df['Volume frame-2'] = np.nan
df['Collagen alignment-1'] = np.nan
df['Collagen alignment-2'] = np.nan

#%% Look back 1-2 frames + save

col_idx = len(df.columns)
df['Neighbor mean dist frame-1'] = np.nan
df['Neighbor mean dist frame-2'] = np.nan
df['Neighbor mean cell volume frame-1'] = np.nan
df['Neighbor mean cell volume frame-2'] = np.nan
df['Neighbor std cell volume frame-1'] = np.nan
df['Neighbor std cell volume frame-2'] = np.nan
df['Neighbor mean FUCCI int frame-1'] = np.nan
df['Neighbor mean FUCCI int frame-2'] = np.nan
df['Neighbor mean height from BM frame-1'] = np.nan
df['Neighbor mean height from BM frame-2'] = np.nan
df['Neighbor max height from BM frame-1'] = np.nan
df['Neighbor max height from BM frame-2'] = np.nan
df['Neighbor mean collagen alignment frame-1'] = np.nan
df['Neighbor mean collagen alignment frame-2'] = np.nan

df['Neighbor planar number frame-1'] = np.nan
df['Neighbor planar number frame-2'] = np.nan
df['Neighbor diff number frame-1'] = np.nan
df['Neighbor diff number frame-2'] = np.nan
# df['Neighbor mean nuclear volume frame-1'] = np.nan # Adding cell vol already
# df['Neighbor mean nuclear volume frame-2'] = np.nan

# Call fate based on cell height
# For each cell, load all frames, then grab the prev frame height data
# Some dynamics
for basalID in collated.keys():
    idx = np.where(df['basalID'] == basalID)[0]
    this_len = len(idx)
    if this_len > 1:
        
        this_cell = df.iloc[idx]
        
        # Cell-instrinsic dynamics from t-1/t-2
        fucci_int = this_cell['FUCCI bg sub'].values
        curvature = this_cell['Mean curvature'].values
        bm_height = this_cell['Height to BM'].values
        vol = this_cell['Volume (sm)'].values
        col_align = this_cell['Collagen alignment'].values
        
        # Environment dynamics from t-1 or t-2
        mean_neighbor_dist = this_cell['Mean neighbor dist'].values
        
        mean_neighbor_vol = this_cell['Mean neighbor cell volume'].values
        std_neighbor_vol = this_cell['Std neighbor cell volume'].values
        
        mean_neighbor_fucci = this_cell['Mean neighbor FUCCI intensity'].values
        
        mean_neighbor_heights = this_cell['Mean neighbor height from BM'].values
        max_neighbor_heights = this_cell['Max neighbor height from BM'].values
        
        mean_neighbor_collagen_alignment = this_cell['Mean neighbor collagen alignment'].values
        
        planar_neighbor_numb = this_cell['Num planar neighbors'].values
        diff_neighbor_numb = this_cell['Num diff neighbors'].values
        
        # Compute d/dt
        # df.at[idx[1:],'Delta curvature'] = np.diff(this_cell['Mean curvature'])
        # df.at[idx[1:],'Delta height'] = np.diff(this_cell['Height to BM'])
        
        # The prev frame
        for t in np.arange(1,this_len):
            # 12h before
            # Intrinsic
            df.at[idx[t],'FUCCI bg sub frame-1'] = fucci_int[t-1]
            df.at[idx[t],'Volume frame-1'] = vol[t-1]
            df.at[idx[t],'Collagen alignment-1'] = col_align[t-1]
            df.at[idx[t],'Delta curvature'] = curvature[t] - curvature[t-1]
            df.at[idx[t],'Delta height'] = bm_height[t] - bm_height[t-1]
            
            # Distance to neighbors
            df.at[idx[t],'Neighbor mean dist frame-1'] = mean_neighbor_dist[t-1]
            
            # Neighbor cell volume
            df.at[idx[t],'Neighbor mean cell volume frame-1'] = mean_neighbor_vol[t-1]
            df.at[idx[t],'Neighbor std cell volume frame-1'] = std_neighbor_vol[t-1]
            
            # Neighbor FUCCI int
            df.at[idx[t],'Neighbor mean FUCCI int frame-1'] = mean_neighbor_fucci[-1]
            
            #Neighbor max+mean neight from BM
            df.at[idx[t],'Neighbor mean height from BM frame-1'] = mean_neighbor_heights[t-1]
            df.at[idx[t],'Neighbor max height from BM frame-1'] = max_neighbor_heights[t-1]
            
            # Neighbor alignment to collagen fibrils
            df.at[idx[t],'Neighbor mean collagen alignment frame-1'] = mean_neighbor_collagen_alignment[t-1]
            
            # Number of neighbors in plane or differentiating
            df.at[idx[t],'Neighbor planar number frame-1'] = planar_neighbor_numb[t-1]
            df.at[idx[t],'Neighbor diff number frame-1'] = diff_neighbor_numb[t-1]
            
            if t > 1:
                # 24h before
                # Intrinsic
                df.at[idx[t],'FUCCI bg sub frame-2'] = fucci_int[t-2]
                df.at[idx[t],'Volume frame-2'] = vol[t-2]
                df.at[idx[t],'Collagen alignment-2'] = col_align[t-2]
                
                # Distance to neighbors
                df.at[idx[t],'Neighbor mean dist frame-2'] = mean_neighbor_dist[t-2]
                
                # Neighbor cell volume
                df.at[idx[t],'Neighbor mean cell volume frame-2'] = mean_neighbor_vol[t-2]
                df.at[idx[t],'Neighbor std cell volume frame-2'] = std_neighbor_vol[t-2]
                
                # Neighbor FUCCI int
                df.at[idx[t],'Neighbor mean FUCCI int frame-2'] = mean_neighbor_fucci[-2]
                
                #Neighbor max+mean neight from BM
                df.at[idx[t],'Neighbor mean height from BM frame-2'] = mean_neighbor_heights[t-2]
                df.at[idx[t],'Neighbor max height from BM frame-2'] = max_neighbor_heights[t-2]
                
                # Neighbor alignment to collagen fibrils
                df.at[idx[t],'Neighbor mean collagen alignment frame-2'] = mean_neighbor_collagen_alignment[t-2]
                
                # Number of neighbors in plane or differentiating
                df.at[idx[t],'Neighbor planar number frame-2'] = planar_neighbor_numb[t-2]
                df.at[idx[t],'Neighbor diff number frame-2'] = diff_neighbor_numb[t-2]
                
        
            
df['NC ratio'] = df['Nuclear volume (sm)']/df['Volume (sm)']
# df['NC ratio raw'] = df['Nuclear volume raw']/df['Volume (sm)']
# df['NC ratio normalized'] = df['Nuclear volume normalized']/df['Volume (sm)']
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

