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

for f in flist[0:1]:
    im = io.imread(f).astype(float)
    print(f'Loaded {f}')
    # Convert all 0 into NaN
    im[ im == 0] = np.nan
    equalized = np.zeros_like(im)
    
    for c in range(im.shape[3]):
        # Run CLAHE
        equalized[:,:,:,c] = equalize_adapthist(this_channel,128)
    print(f'Equalized {f}')

    # save
    io.imsave(path.splitext(f)[0] + '_eq.tif', util.img_as_uint(equalized))
    print(f'Saved {f}')

