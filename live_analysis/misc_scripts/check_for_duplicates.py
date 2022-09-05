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

for t in range(15):
    im = io.imread(path.join(dirname,f'3d_nuc_seg/naive_tracking/t{t}.tif'))
    
    df_ = pd.DataFrame(measure.regionprops_table(im,properties=['label','euler_number']))
    
    badIDs = df_[df_['euler_number'] > 1]['label']
    for l in badIDs:
        mask = im == l
        if measure.label(mask).max() > 1:
            print(f't = {t}, label = {l}')
    
