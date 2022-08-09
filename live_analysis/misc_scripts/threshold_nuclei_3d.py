#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 23:05:43 2022

@author: xies
"""

from skimage import io,filters,morphology
import numpy as np
import os.path as path
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-08-2022/F2 WT/R2/H5/'

h2b = io.imread(path.join(dirname,'master_stack.tif'))[...,1]

#%%
T,Z,X,Y = h2b.shape
#
mask = np.zeros_like(h2b)

for t in tqdm(range(T)):
    for z in tqdm(range(Z)):
        im = h2b[t,z,...]
        if im.sum() == 0:
            continue
        
        im_ = im.astype(float)
        im_[im == 0] = np.nan
        # Apply local threshold with radius = 31        
        radius = 31
        # selem = morphology.disk(radius)
        th = filters.threshold_local(im_,block_size=35)
        mask[t,z,...] = im >= th
        # mask[t,z,...] = filters.rank.threshold(im, selem)
        # entropy = filters.rank.threshold_percentile(im, selem,p0=0.5)
        # mask[t,z,...] = (entropy >= 800).astype(np.int8)
        
#%%

#mask = io.imread(path.join(dirname,'h2b_mask.tif'))
#
# Do some clean up
for t in tqdm(range(T)):
    for z in range(Z):
        m = mask[t,z,...]
        m = morphology.binary_opening(m)
        m = morphology.binary_closing(m)
        m = binary_fill_holes(m)
        mask[t,z,...] = m

io.imsave(path.join(dirname,'h2b_mask_clean.tif'),mask.astype(np.int8))

