#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:05:41 2024

@author: xies
"""

import numpy as np
from skimage import io, measure
from os import path
import pandas as pd
from scipy import spatial
from tqdm import tqdm
import matplotlib.pyplot as plt

dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/diTalia_zebrafish/osx_fucci_26hpp_11_4_17/stardist/'

img = io.imread(path.join(dirname,'training_images/patch1_t65.tif'))
label_stack = io.imread(path.join(dirname,'training_labels/patch1_t65.tif'))

df = []
for z, im in enumerate(label_stack):
    _df = pd.DataFrame(measure.regionprops_table(im,properties=['area','label','centroid']))
    _df['Z'] = z
    df.append(_df)
    
df = pd.concat(df,ignore_index=True)
df = df.rename(columns={'centroid-0':'Y','centroid-1':'X'})

#%% Calculate pairwise distances

df_grouped = [_df for _,_df in df.groupby('Z')]
z_grouped = [z for z,_ in df.groupby('Z')]
next_z_paired = zip(z_grouped[:-1],z_grouped[1:])
next_df_paired = zip(df_grouped[:-1],df_grouped[1:])

distMats = {}
for (z1,z2),(x,y) in zip(next_z_paired,next_df_paired):
    x = x[['X','Y']]
    y = y[['X','Y']]
    D = spatial.distance_matrix(x, y)
    distMats[(z1,z2)] = D

#%%

nearest_label_downward = {}
for (z1,z2),D in distMats.items():
    min_idx = np.argmin(D,axis=1).astype(float)
    min_idx[D.min(axis=1) > 20] = np.nan
    nearest_label_downward[(z1,z2)] = min_idx

#%%

def overlap(mask1,mask2):
    and_mask = np.logical_and(mask1,mask2)
    return and_mask.sum() / mask1.sum()

for (z1,z2),candidate in nearest_label_downward.items():
    
    for i in range((df['Z'] == z1).sum()):
        this_mask = label_stack[z1,:] == i
        downward_candidate = label_stack[z1,:] == candidate[i]
        if overlap(this_mask,downward_candidate) < 0.5:
            nearest_label_downward[(z1,z2)][i] = np.nan
            
#%%

stitched_mask
for (z1,z2)




