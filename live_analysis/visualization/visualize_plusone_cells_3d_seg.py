#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:36:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io,util
from os import path
import pickle as pkl
from tqdm import tqdm
 
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
df = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)

#%%

missing = []
# for basalID in tqdm(np.unique|(df['basalID'])):
for basalID in [374]:
    
    cell = df[df['basalID'] == basalID]
    cell = cell.sort_values('Frame')
    
    # Find the relevant frames
    T = cell['Frame'].values
    central_cellposeID = {}
    neighbor_cellposeIDs = {}
    central_cytoID = {}
    neighbor_cytoIDs = {}
    for t in T:
        
        cell_current_frame = cell[cell['Frame'] == t]
        central_cellposeID[t] = cell_current_frame['CellposeID'].iloc[0]
        central_cytoID[t] = cell_current_frame['CytoID'].iloc[0]
        
        # Load adjacency mask
        adj_dict = np.load(path.join(dirname,f'Image flattening/flat_adj_dict/adjdict_t{t}.npy'),allow_pickle=True).item()
        
        neighbor_cellposeIDs[t] = adj_dict[central_cellposeID[t]]
        
        # Load cyto cellpose
        cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
        
        df_this_frame = df[df['Frame'] == t]
        neighbors = df_this_frame[df_this_frame['CellposeID'].isin(neighbor_cellposeIDs[t])]
        neighbor_cytoIDs[t] = neighbors['CytoID'].values
        # Report missing CytoID segmentations
        missing.append(neighbors[(neighbors['CytoID'] == 0) | np.isnan(neighbors['CytoID'])])

missing = pd.concat(missing,ignore_index=True)

#%%

t = 1

print('---')
missing = missing.sort_values('CellposeID')
for i,neighb in missing[missing['Frame'] == t].iterrows():
    print(f'Missing cyto_seg: frame = {t}, CellposeID = {neighb.CellposeID}')


#%% Visualize

CROP = False

basalID = 374
cell = df[df['basalID'] == basalID]
T = cell['Frame'].values

# Generate full masks
if CROP:
    TT = len(T)
else:
    TT = 15
center_mask = np.zeros((TT,*cyto_seg.shape),dtype=int)
neighbor_mask = np.zeros((TT,*cyto_seg.shape),dtype=int)
ticker = 2
for i,t in enumerate(T):
    cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
    
    if CROP:
        idx = i
    else:
        idx = t
    center_mask[idx,cyto_seg == central_cytoID[t]] = basalID
    # neighbor_mask[idx,...] = center_mask[idx,...]
    for neighborID in neighbor_cytoIDs[t]:
        neighbor_mask[idx,cyto_seg == neighborID] = ticker
        ticker += 1
    
if CROP:
    # Trim movie down to size by cropping out blank space
    border_xy = 50 #px on edge
    border_z = 2 # px up and down
    
    It,Iz,Iy,Ix = np.where(neighbor_mask + center_mask)
    Ztop = Iz.min() - border_z
    Zbottom = Iz.max() + border_z
    
    Ymin = Iy.min () - border_xy
    Xmin = Ix.min () - border_xy
    Ymax = Iy.max () + border_xy
    Xmax = Ix.max () + border_xy
    
    center_mask = center_mask[:,Ztop:Zbottom,Ymin:Ymax,Xmin:Xmax]
    neighbor_mask = neighbor_mask[:,Ztop:Zbottom,Ymin:Ymax,Xmin:Xmax]
    
    io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/center-crop.tif'),util.img_as_uint(center_mask))
    io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/neighbor-crop.tif'),util.img_as_uint(neighbor_mask))

else:
    io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/center-full.tif'),util.img_as_uint(center_mask))
    io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/neighbor-full.tif'),util.img_as_uint(neighbor_mask))



