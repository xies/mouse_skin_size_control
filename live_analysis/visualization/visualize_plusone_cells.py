#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:36:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io
from os import path
import pickle as pkl
from tqdm import tqdm
 
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)

#%%

missing = []
for basalID in tqdm(np.unique(df['basalID'])):
    cell = df[df['basalID'] == basalID]
    cell = cell.sort_values('Frame')
    
    # Find the relevant frames
    T = cell['Frame'].values
    0
    central_cellposeID = {}
    neighbor_cellposeIDs = {}
    for t in T:
        
        cell_current_frame = cell[cell['Frame'] == t]
        central_cellposeID[t] = cell_current_frame['CellposeID'].iloc[0]
        
        # Load adjacency mask
        adj_dict = np.load(path.join(dirname,f'Image flattening/flat_adj_dict/adjdict_t{t}.npy'),allow_pickle=True).item()
        
        neighbor_cellposeIDs[t] = adj_dict[central_cellposeID[t]]
        
        # Load cyto cellpose
        cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
        
        df_this_frame = df[df['Frame'] == t]
        neighbors = df_this_frame[df_this_frame['CellposeID'].isin(neighbor_cellposeIDs[t])]
        # Report missing CytoID segmentations
        missing.append(neighbors[(neighbors['CytoID'] == 0) | np.isnan(neighbors['CytoID'])])

missing = pd.concat(missing,ignore_index=True)

#%%

t = 1

print('---')
missing = missing.sort_values('CellposeID')
for i,neighb in missing[missing['Frame'] == t].iterrows():
    print(f'Missing cyto_seg: frame = {t}, CellposeID = {neighb.CellposeID}')



