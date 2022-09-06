#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:10:07 2022

@author: xies
"""

from skimage import io, measure
import numpy as np
import pandas as pd

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%%

# Clean up small objects
for t in range(15):
    im = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    
    df_ = pd.DataFrame(measure.regionprops_table(im,properties=['label','area']))
        
    # clean up again for area (some speckles are left behind)
    df_ = df_[df_['area'] > 400]
    
    im[~np.in1d(im.flatten(),df_['label']).reshape(im.shape)] = 0

    io.imsave(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'), im)

#%%

for t in range(15):
    
    # Also check for size range
    
    # im = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}.tif'))
    im = io.imread(path.join(dirname,f'3d_nuc_seg/naive_tracking/t{t}.tif'))
    
    df_ = pd.DataFrame(measure.regionprops_table(im,properties=['label','euler_number','area']))
    
    badIDs = df_[df_['euler_number'] > 1]['label']
    for l in badIDs:
        mask = im == l
        if measure.label(mask).max() > 1:
            print(f'Duplicate: t = {t}, label = {l}')
    
    badIDs = df_[df_['area'] < 500]['label']
    
    print(f'Too small: t = {t}, {badIDs}')
    
    