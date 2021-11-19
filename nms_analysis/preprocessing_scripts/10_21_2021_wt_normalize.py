#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:21:12 2021

@author: xies
"""

import numpy as np
from skimage import io, util
from skimage.exposure import equalize_adapthist
from os import path
from glob import glob

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/RB-KO/R3'

#%%

flist = glob(path.join(dirname,'*/*_c.tif'))


#%% Normalize

for f in flist:
    im = io.imread(f).astype(float)
    print(f'Loaded {f}')
    # Convert all 0 into NaN
    im[ im == 0] = np.nan
    
    equalized = np.zeros_like(im)
    
    for c in range(im.shape[3]):
        
        this_channel = im[...,c]
        # Rectify to non-NaN zero
        min_int = np.nanmin(this_channel)
        this_channel[ np.isnan(this_channel) ] = 0
        this_channel -= min_int
        this_channel[ this_channel < 0 ] = 0
        
        # Run CLAHE slice-wize
        for z in range(im.shape[0]):
            equalized[z,:,:,c] = equalize_adapthist(this_channel[z,...].astype(np.int16),256)
            
    print(f'Equalized {f}')

    # save
    io.imsave(path.splitext(f)[0] + '_eq.tif', util.img_as_uint(equalized))
    print(f'Saved {f}')

