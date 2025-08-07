#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:29:50 2022

@author: xies
"""

import numpy as np
from skimage import io, measure
from glob import glob
from os import path
from tqdm import tqdm
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.path import Path
from SelectFromCollection import SelectFromCollection

'''

1. Load cellpose output on whole z stack
2. Use heightmap and identify the likely 'basal z-range' and get rid of things more apical
3. get rid of small + large objects

'''

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Ablation time courses/M1 M2 K14 Rbfl DOB 06-01-2023/01-13-2024 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/R1'
# dirname = '/Volumes/T7/01-13-2023 Ablation K14Cre H2B FUCCI/Black right clipped DOB 06-30-2023/R1'
 
#%%

T = 1

# predictions = io.imread(path.join(dirname,'im_seq_decon/t2_decon_masks.tif'))
# heightmaps = io.imread(path.join(dirname,'im_seq_decon/t2_height_map.tif'))
SEG_DIR = 'cellpose_G_clahe_blur'
FLAT_DIR = 'Image flattening/heightmaps'

# Some pruning parameters
# MIN_SIZE_IN_PX = 2000
_tmp = []
for t in tqdm(range(T)):
    
    predictions = io.imread(path.join(dirname,f'{SEG_DIR}/t{t}_3d_nuc/t{t}_masks.tif'))
    heightmaps = io.imread(path.join(dirname,f'{FLAT_DIR}/t{t}.tif'))
    
    table = pd.DataFrame(measure.regionprops_table(predictions,properties={'label','area','centroid','bbox'}))
    
    # Look at each XY coord and look up heightmap
    Z = heightmaps[ table['centroid-1'].round().astype(int), table['centroid-1'].round().astype(int) ]
    table['Corrected Z'] = table['bbox-0'] - Z
    
    table['Time'] = t
    _tmp.append(table)
    
#%%%

# ts = ax.scatter(grid_x, grid_y)

df = pd.concat(_tmp)
# df = pd.concat([wt_table,rbko_table])

plt.figure()

pts = plt.scatter(df['area'],df['Corrected Z'],alpha=0.01)
plt.ylabel('Corrected Z (to heightmap)')
plt.xlabel('Cell size (fL)')
# plt.xlim([0,25000])
# gate = roipoly()

selector = SelectFromCollection(plt.gca(), pts)

#%% Gate the cells

verts = np.array(selector.poly.verts)
x = verts[:,0]
y = verts[:,1]

p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['area'],df['Corrected Z'])])
# I = p_.contains_points( np.array([df['area'],df['Corrected Z']]).T )

df_ = df[I]
# plt.scatter(df['area'],df['centroid-0'],alpha=0.5)
# plt.scatter(df_['area'],df_['centroid-0'],color='r')

#%% # Reconstruct the filtered segmentation predictions

OUT_SUBDIR = 'cellpose_clean'

for t in tqdm(range(T)):
    
    predictions = io.imread(path.join(dirname,f'{SEG_DIR}/t{t}_3d_nuc/t{t}_masks.tif'))
    
    this_cellIDs = df_[df_['Time'] == t]['label']
    
    filtered_pred = predictions.flatten()
    I = np.in1d(predictions.flatten(), this_cellIDs)
    filtered_pred[~I] = 0
    filtered_pred = filtered_pred.reshape(predictions.shape)
    
    io.imsave(path.join(dirname,f'{OUT_SUBDIR}/t{t}_clean.tif'),filtered_pred.astype(np.int16))
    # io.imsave(path.join(dirname,f'RBKO1_nuc_seg_cleaned.tif'),filtered_pred.astype(np.int16))
        



