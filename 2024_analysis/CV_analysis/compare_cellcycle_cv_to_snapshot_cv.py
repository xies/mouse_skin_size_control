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
from mathUtils import cvariation_ci, cvariation_ci_bootstrap

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

#%% CV: parametric CI estimates

cv_g1,g1_lb,g1_ub = cvariation_ci(cycling_g1['Volume'])
cv_sg2,sg2_lb,sg2_ub = cvariation_ci(cycling_sg2['Volume'])

print('---- Cycling skin cells: pooling all time points ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {g1_lb:.3}, upper bound: {g1_ub:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {sg2_lb:.3}, upper bound: {sg2_ub:.3}")


cv_g1,g1_lb,g1_ub = cvariation_ci(dense_g1['Cell volume'])
cv_sg2,sg2_lb,sg2_ub = cvariation_ci(dense_sg2['Cell volume'])

print('---- All cells ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {g1_lb:.3}, upper bound: {g1_ub:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {sg2_lb:.3}, upper bound: {sg2_ub:.3}")

#%% CV: bootstrap CI estimates

Nboot = 1000
cv_g1,g1_lb,g1_ub = cvariation_ci_bootstrap(cycling_g1['Volume'],Nboot)
cv_sg2,sg2_lb,sg2_ub = cvariation_ci_bootstrap(cycling_sg2['Volume'],Nboot)

print('---- Cycling skin cells: pooling all time points ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {g1_lb:.3}, upper bound: {g1_ub:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {sg2_lb:.3}, upper bound: {sg2_ub:.3}")


cv_g1,g1_lb,g1_ub = cvariation_ci_bootstrap(dense_g1['Cell volume'],Nboot)
cv_sg2,sg2_lb,sg2_ub = cvariation_ci_bootstrap(dense_sg2['Cell volume'],Nboot)

print('---- All cells ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {g1_lb:.3}, upper bound: {g1_ub:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {sg2_lb:.3}, upper bound: {sg2_ub:.3}")



