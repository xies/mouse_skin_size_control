#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 06:14:24 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import seaborn as sb

from scipy.stats import stats
import matplotlib.pyplot as plt

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%% Load complete cell cycles and snapshots

complete_cycles = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
(_,(_,cycling_g1),(_,cycling_sg2)) = complete_cycles.groupby('Phase')

# Load segmentation
nuc_seg = io.imread(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t0.tif'))

# Load manual annotations for snapshots
g1_anno = pd.read_csv(path.join(dirname,'dense_cellcycle_annotations/t0_g1.csv'),index_col=0)
# g1_anno = pd.concat((g1_anno,pd.read_csv(path.join(dirname,'dense_cellcycle_annotations/t0_g1_falseneg.csv'),index_col=0)),ignore_index=True)
g1_anno = g1_anno.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
g1_anno = np.round(g1_anno).astype(int)

# Load dense measurements
dense = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)
dense = dense[dense['Frame'] == 0]

# Exclude cells based on lack of all FUCCI signal
# dense['Exclude'] = False
# exclude = pd.read_csv(path.join(dirname,'dense_cellcycle_annotations/t0_exclude.csv'),index_col=0)
# exclude = exclude.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
# exclude = np.round(exclude).astype(int)
# for _,anno in exclude.iterrows():
#     ID = nuc_seg[anno['Z'],anno['Y'],anno['X']]
#     if ID > 0:
#         dense['Exclude'] = True
        
dense = dense[~dense.Border]
# dense = dense[dense.Exclude]

# transfer annotation into segmentation
g1_cellposeIDs = []
for _,anno in g1_anno.iterrows():
    ID = nuc_seg[anno['Z'],anno['Y'],anno['X']]
    if ID > 0:
        g1_cellposeIDs.append(ID)
g1_cellposeIDs = np.array(g1_cellposeIDs)

g1_segs = np.zeros_like(nuc_seg)
dense['Phase'] = 'SG2'
for ID in g1_cellposeIDs:
    dense.loc[dense['CellposeID'] == ID,'Phase'] = 'G1'
    g1_segs[nuc_seg == ID] = 1
    
((_,dense_g1),(_,dense_sg2)) = dense.groupby('Phase')
io.imsave('/Users/xies/Desktop/g1_segs.tif',g1_segs)

#%%

cv_g1_complete = cycling_g1['Volume'].std() / cycling_g1['Volume'].mean()
cv_sg2_complete = cycling_sg2['Volume'].std() / cycling_sg2['Volume'].mean()

cv_g1_dense = dense_g1['Cell volume'].std() / dense_g1['Cell volume'].mean()
cv_sg2_dense = dense_sg2['Cell volume'].std() / dense_sg2['Cell volume'].mean()

