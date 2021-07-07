np,#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:46:59 2021

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, transform, util

from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/06-27-2021 WT/M6 No Tam/Day 2.5/'

#%%

im = io.imread(path.join(dirname,'B_reg.tif'))

# Go through all z positions and find rows/cols that are all 0
x_zeros = [ np.where(np.all( z_slice == 0 , axis=0))[0] for z_slice in im]
y_zeros = [ np.where(np.all( z_slice == 0 , axis=1))[0] for z_slice in im]


# See if left or right shifted
x_transl = np.zeros( im.shape[0] , dtype=int)
for i,row in enumerate(x_zeros):
    if len(row) == 0:
        continue
    if np.any( row == 0 ):
        shift = -)max(row)+1)
    else:
        shift = 1023-min(row)
    x_transl[i] = shift


# See if up or down shifted
y_transl = np.zeros( im.shape[0] , dtype=int)
for i,row in enumerate(y_zeros):
    if len(row) == 0:
        continue
    if np.any( row == 0 ):
        shift = -(max(row)+1)
    else:
        shift = 1023-min(row)
    y_transl[i] = shift

#%% Transform other image

im_G = io.imread(path.join(dirname,'G.tif'))

G_reg = np.zeros_like(im_G, dtype=np.float)
translations = zip(x_transl,y_transl)

for i,transl in enumerate(translations):
    
    T = transform.SimilarityTransform(translation = transl)
    G_reg[i,...] = transform.warp(im_G[i,...], T)
                 
io.imsave(path.join('/Users/xies/Desktop/','G_reg.tif'),util.img_as_uint(G_reg))