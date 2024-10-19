#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:52:56 2024

@author: xies
"""

import pandas as pd
import numpy as np
from os import path
import pickle as pkl

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'),index_col=0)

def gradient_with_nan(y,edge_order):
    I = ~np.isnan(y)
    dy = np.ones(len(y))*np.nan
    if I.sum() > 1:
        x = np.arange(len(y))
        dy[I] = np.gradient(y[I], x[I], edge_order=edge_order)
    return dy

#%%

all_neighbor_cellID = []
for t in range(45):
    with open(path.join(dirname,f'geodesic_neighbors/geodesic_neighbors_T{t+1:04d}.pkl'),'rb') as f:
        all_neighbor_cellID.append(pkl.load(f))
        
#%%

df_by_frame = {frame:_df for frame,_df in df.groupby('Frame')}
tracks = {ID:t for ID,t in df.groupby('trackID')}

for trackID,track in tracks.items():
    
    # Smooth the relevant things
    track['Change in local cell density'] = gradient_with_nan(track['Local cell density'],2)
    track['Change in mean neighbor Cdt1'] = gradient_with_nan(track['Mean neighbor Cdt1 intensity'],2)
    track['Change in mean neighbor Geminin'] = gradient_with_nan(track['Mean neighbor Gem intensity'],2)
    track['Change in Cdt1'] = gradient_with_nan(track['Mean Cdt1 intensity'],2)
    track['Change in Geminin'] = gradient_with_nan(track['Mean Gem intensity'],2)
    
    track['Change in mean neighbor size'] = gradient_with_nan(track['Mean neighbor volume'],2)
    track['Change in std neighbor size'] = gradient_with_nan(track['Std neighbor volume'],2)
    
# Collapse df again
df_combined = pd.concat(tracks,ignore_index=True)
df_combined.to_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'))

#%%