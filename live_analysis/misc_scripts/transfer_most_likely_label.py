#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:44:34 2023

@author: xies
"""

import numpy as np
from os import path
from tqdm import tqdm
from skimage import io, measure
import pandas as pd
from imageUtils import most_likely_label
import matplotlib.pyplot as plt

# seed_dir = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1/cellpose_low_pass/cellpose_manual'
# target_dir = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1/cellpose_clahe/'

seed_dir = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/3d_nuc_seg/cellpose_cleaned_manual'
target_dir = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/cellpose_clahe'

#%%

df = []
for t in tqdm(range(15)):
    
    seed_labels = io.imread(path.join(seed_dir,f't{t}.tif'))
    target_labels = io.imread(path.join(target_dir,f't{t}_3d_nuc/t{t}_masks.tif'))
    
    _tmp = pd.DataFrame( measure.regionprops_table(seed_labels, intensity_image = target_labels
                                                   ,properties=['label','area']
                                                   ,extra_properties = [most_likely_label]))
    _tmp = _tmp[_tmp['area'] > 1000]
    _tmp['Frame'] = t
    df.append(_tmp)

df = pd.concat(df,ignore_index=True)

#%% Filter for real cells

# plt.hist(df['area'],1000)

# cutoff = 1000

# df = df[df['area'] > cutoff]

#%%
for t in tqdm(range(15)):
    
    target_labels = io.imread(path.join(target_dir,f't{t}_3d_nuc/t{t}_masks.tif'))
    
    mask = np.zeros_like(target_labels,dtype=bool)
    
    df_ = df[df['Frame'] == t]
    for l in tqdm(df_['most_likely_label'].values):
        mask = mask | (target_labels == l)
        
    transferred_labels = target_labels * mask
    
    io.imsave(path.join(target_dir,f'manual_transfer/t{t}.tif'),transferred_labels)
    
    