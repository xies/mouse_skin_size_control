#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:05:02 2022

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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-168hr_R2.tif'))
dirname = '/Users/xies/OneDrive - Stanford/Skin/Confocal/08-26-2022/10month 2week induce/Paw H2B-CFP FUCCI2 Phall647/WT1'
fname = 'WT1'
filenames = glob(path.join(dirname,'{fname}.tif'))

#%%

# pred_name = 'cyto_masks'
pred_name = 'nuc_masks'

# Some pruning parameters
# MIN_SIZE_IN_PX = 2000

_tmp = []

predictions = io.imread(path.join(dirname,f'{fname}_{pred_name}.tif'))
heightmaps = io.imread(path.join(dirname,f'Image flattening/heightmaps/{fname}.tif'))
table = pd.DataFrame(measure.regionprops_table(predictions,properties={'label','area','centroid','bbox'}))

# Look at each XY coord and look up heightmap
Z = heightmaps[ table['centroid-1'].round().astype(int), table['centroid-1'].round().astype(int) ]
table['Corrected Z'] = table['bbox-0'] - Z

table['Time'] = t
_tmp.append(table)
    
#%%%

# ts = ax.scatter(grid_x, grid_y)

df = pd.concat(_tmp)

plt.figure()

pts = plt.scatter(df['area'],df['Corrected Z'],alpha=0.1)
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

# OUT_NAME = '_cyto_seg_cleaned'
OUT_NAME = '_nuc_seg_cleaned'

this_cellIDs = df_[df_['Time'] == t]['label']

filtered_pred = predictions.flatten()
I = np.in1d(predictions.flatten(), this_cellIDs)
filtered_pred[~I] = 0
filtered_pred = filtered_pred.reshape(predictions.shape)

io.imsave(path.join(dirname,f'{fname}{OUT_NAME}.tif'),filtered_pred.astype(np.uint16))
    


