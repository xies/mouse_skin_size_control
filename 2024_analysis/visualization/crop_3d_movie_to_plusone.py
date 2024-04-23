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

#%% Load the adjacencies

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

#%% Visualize

basalID = 374
cell = df[df['basalID'] == basalID]

# Load the other channels as well
R = io.imread(path.join(dirname,'Cropped_Images/R.tif'))
G = io.imread(path.join(dirname,'Cropped_Images/G.tif'))
B = io.imread(path.join(dirname,'Cropped_Images/B.tif'))
basal_cyto = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_cyto.tif'))

# Generate full masks
TT = 15

neighbor_mask = np.zeros((TT,*cyto_seg.shape),dtype=int)
for i,t in enumerate(T):
    
    cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
    
    idx = t
    # neighbor_mask[idx,...] = center_mask[idx,...]
    for neighborID in neighbor_cytoIDs[t]:
        neighbor_mask[idx,cyto_seg == neighborID] = 1
    
    
# Trim movie down to size using dimensions of the neighbors mask
border_xy = 10 #px on edge

It,Iz,Iy,Ix = np.where(neighbor_mask)

Ymin = Iy.min () - border_xy
Xmin = Ix.min () - border_xy
Ymax = Iy.max () + border_xy
Xmax = Ix.max () + border_xy

cropped = np.zeros((6,TT,72,Ymax-Ymin,Xmax-Xmin))
cropped_adj_image = np.zeros((TT,Ymax-Ymin,Xmax-Xmin))
for t in range(TT):
    
    cyto_seg = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
    nuc_seg = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    
    
    adj_image = io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t{t}.tif'))
    
    cropped[0,t,:,:,:] = R[t,:,Ymin:Ymax,Xmin:Xmax]
    cropped[1,t,:,:,:] = G[t,:,Ymin:Ymax,Xmin:Xmax]
    cropped[2,t,:,:,:] = B[t,:,Ymin:Ymax,Xmin:Xmax]
    cropped[3,t,:,:,:] = nuc_seg[:,Ymin:Ymax,Xmin:Xmax]
    cropped[4,t,:,:,:] = cyto_seg[:,Ymin:Ymax,Xmin:Xmax]
    cropped[5,t,:,:,:] = basal_cyto[t,:,Ymin:Ymax,Xmin:Xmax]
    
    cropped_adj_image[t,...] = adj_image[Ymin:Ymax,Xmin:Xmax]

io.imsave(path.join('/Users/xies/Desktop/example_mouse_skin_image.tif'),cropped.astype(np.uint16))
io.imsave(path.join('/Users/xies/Desktop/example_mouse_skin_cell_contact_map.tif'),cropped_adj_image.astype(np.uint16))


#     io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/center-crop.tif'),util.img_as_uint(center_mask))
#     io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/neighbor-crop.tif'),util.img_as_uint(neighbor_mask))

# # else:
#     io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/center-full.tif'),util.img_as_uint(center_mask))
#     io.imsave(path.join(dirname,f'Examples for figures/Microenvironment movies/{basalID}/neighbor-full.tif'),util.img_as_uint(neighbor_mask))



