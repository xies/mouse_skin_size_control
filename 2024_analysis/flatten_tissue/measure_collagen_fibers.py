#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 00:06:19 2022

@author: xies

NB: Probably more useful to calculate
    
"""

from skimage import io, filters,util
import numpy as np
import pandas as pd

import scipy.ndimage as ndi

from os import path
from glob import glob
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# imfiles = glob(path.join(dirname,'Image flattening/flat_z_shift_2/t*.tif'))

imstack = io.imread(path.join(dirname,'Cropped_images/B.tif'))

XX = 460
Z = 72
T = 15

#%%

TOP_OFFSET = 2
BOTTOM_OFFSET = 8

for t in tqdm(range(T)):
    
    # im = io.imread(filenames[t])
    im = imstack[t,...]
    Z,XX,_ = im.shape
    
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    output_dir = path.join(dirname,'Image flattening/flat_collagen')
    
    flat = np.zeros((XX,XX))
    Iz_top = heightmap + TOP_OFFSET
    Iz_bottom = heightmap + BOTTOM_OFFSET
    
    for x in range(XX):
        for y in range(XX):
            
            # flat_indices = np.arange(0,TOP_OFFSET-BOTTOM_OFFSET)
            
            z_coords = np.arange(Iz_top[y,x],Iz_bottom[y,x])
            # sanitize for out-of-bounds
            z_coords[z_coords < 0] = 0
            z_coords[z_coords >= Z] = Z-1
            
            I = (z_coords > 0) & (z_coords < Z)
            
            flat[y,x] = im[z_coords[I],y,x].mean(axis=0)
    
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16),check_contrast=False)
    
    