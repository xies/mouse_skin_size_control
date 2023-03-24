#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:36:03 2022

@author: xies
"""


import numpy as np
from skimage import io
from glob import glob
from os import path
from tqdm import tqdm

from re import match

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

def sort_by_timestamp(filename):
    t = match('t(\d+).tif',filename).groups[0]
    return int(t)

filenames = sorted(glob(path.join(dirname,'im_seq/t*.tif')),key=sort_by_timestamp)

#%% Load a heightmap and flatten the given z-stack

BOTTOM_OFFSET = 5
TOP_OFFSET = -20

imstack = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))
T,Z,XX,_ = imstack.shape

# im2flatten = imstack[...,1] # flatten green channel
# im2flatten = imstack

for t in range(15):
    
    im = io.imread(path.join(dirname,f'im_seq/t{t}.tif'))
    im = imstack[t,...]
    
    # heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    output_dir = path.join(dirname,f'Image flattening/flat_3d_nuc_seg')
    
    flat = np.zeros((BOTTOM_OFFSET-TOP_OFFSET,XX,XX))
    Iz_top = heightmap + TOP_OFFSET
    Iz_bottom = heightmap + BOTTOM_OFFSET
    for x in range(XX):
        for y in range(XX):
            
            flat[:,y,x] = im[Iz_top[y,x]:Iz_bottom[y,x],y,x]
                
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    
        