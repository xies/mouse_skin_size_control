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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'),index_col=0)

#%%

all_neighbor_idx = []
for t in range(65):
    with open(path.join(dirname,f'geodesic_neighbors/geodesic_neighbors_dfindex_T{t+1:04d}.pkl'),'rb') as f:
        all_neighbor_idx.append(pkl.load(f))
        

#%%

df_by_frame = {frame:_df for frame,_df in df.groupby('Frame')}
tracks = {ID:t for ID,t in df.groupby('trackID')}

for trackID,track in tracks.items():
    
    # Smooth the relevant things
    df['Change in local cell density'] = np.gradient(df['Local cell density'],2)
    df['Change in mean neighbor Cdt1'] = np.gradient(df['Mean neighbor Cdt1'],2)
    df['Change in mean neighbor Geminin'] = np.gradient(df['Mean neighbor Gem'],2)
    df['Change in Cdt1'] = np.gradient(df['Mean Cdt1 intensity'],2)
    df['Change in Geminin'] = np.gradient(df['Mean Gem intensity'],2)
    
    
    
    