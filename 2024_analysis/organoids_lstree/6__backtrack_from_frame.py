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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 31_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'),index_col=0)
df['organoidID'] = 6
df['organoidID_trackID'] = df['organoidID'].astype(str) + '_' + df['trackID'].astype(str)

def gradient_with_nan(y,edge_order):
    I = ~np.isnan(y)
    dy = np.ones(len(y))*np.nan
    if I.sum() > edge_order + 1:
        x = np.arange(len(y))
        dy[I] = np.gradient(y[I], x[I], edge_order=edge_order)
    return dy

all_neighbor_cellID = []
for t in range(50):
    with open(path.join(dirname,f'geometric_neighbors/geometric_neighbors_dfindex_T{t+1:04d}.pkl'),'rb') as f:
        all_neighbor_cellID.append(pkl.load(f))
        
#%%

tracks = {ID:t for ID,t in df.groupby('organoidID_trackID')}
# del tracks['5_nan']
# del tracks['2_nan']
del tracks['6_nan']

for trackID,track in tracks.items():
    
    # Smooth the relevant things
    track['Change in local cell density'] = gradient_with_nan(track['Local cell density'],2)
    track['Change in mean neighbor Cdt1'] = gradient_with_nan(track['Mean neighbor Cdt1 intensity'],2)
    track['Change in mean neighbor Geminin'] = gradient_with_nan(track['Mean neighbor Gem intensity'],2)
    track['Change in Cdt1'] = gradient_with_nan(track['Normalized Cdt1 intensity'],2)
    track['Change in Geminin'] = gradient_with_nan(track['Normalized Gem intensity'],2)
    
    track['Change in mean neighbor size'] = gradient_with_nan(track['Mean neighbor volume'],2)
    track['Change in std neighbor size'] = gradient_with_nan(track['Std neighbor volume'],2)
    
# Collapse df again
df_combined = pd.concat(tracks,ignore_index=True)

#%%  Automatically 'call' G1/S

diffs = {}
for trackID,track in tracks.items():
    
    criterion = track['Normalized Cdt1 intensity'].argmax()
    # if track.iloc[0].trackID == 45:
    #     stop
    track['Frame with largest Cdt1 decrease'] = track.iloc[criterion]['Frame']
    track['Frame since birth with largest Cdt1 decrease'] = criterion
    
    # Redo phase
    auto_phase = np.zeros(len(track))
    
    I = np.where(track.Phase =='G1S')[0]
    if I.sum()>0:
        track['First manual G1S frame'] = I[0]
        diffs[trackID] = track['Change in Cdt1'].argmin() - I[0]
        auto_phase[criterion:] = 1
    track['Auto phase'] = auto_phase == 1
    
    
df_combined = pd.concat(tracks,ignore_index=True)
df_combined.to_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'))

#%%

