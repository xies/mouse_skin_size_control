#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:33:19 2024

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

all_basalIDs = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_cyto.tif'))
manual_tracking = io.imread(path.join(dirname,f'manual_basal_tracking/basal_tracks_cyto.tif'))


for t in range(15):
    
    # Load cytoID
    cytoIDs = io.imread(path.join(dirname,f'3d_cyto_seg/3d_cyto_manual/t{t}_cleaned.tif'))
    #nucID
    nucIDs = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    #bnasalIDs
    basalIDs = all_basalIDs[t,...]
    
    this_df = df[df['Frame'] == t]
    
    new_cytoIDs = np.zeros_like(cytoIDs)
    
    all_cytoIDs = np.unique(cytoIDs)[1:]
    for cID in all_cytoIDs:
        if (this_df['CytoID'] == cID).sum() > 0:
            new_label = this_df[this_df['CytoID'] == cID]['CellposeID'].iloc[0]
            if not np.isnan(new_label) and new_label > 0:
                new_cytoIDs[cytoIDs == cID] = new_label
    
    # If basalID present, overwrite mask entirely
    all_cytoIDs = np.unique(cytoIDs)[1:]
    for cID in all_cytoIDs:
        if (this_df['CytoID'] == cID).sum() > 0:
            new_label = this_df[this_df['CytoID'] == cID]['basalID'].iloc[0]
            if not np.isnan(new_label) and new_label > 0:
                new_cytoIDs[manual_tracking[t,...] == new_label] = new_label

    io.imsave(path.join(dirname,f'3d_cyto_seg/nucID_label_transfered/t{t}.tif'), new_cytoIDs)
              
              