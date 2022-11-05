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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = glob(path.join(dirname,'im_seq/t*.tif'))

#%% Load a heightmap and flatten the given z-stack

BOTTOM_OFFSET = 5
TOP_OFFSET = -20


imstack = io.imread(filenames[0])
T,Z,XX,_ = imstack.shape

im2flatten = imstack[...,1] # flatten green channel
# im2flatten = imstack

for t,im in tqdm(enumerate(im2flatten)):
    
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    output_dir = path.join(dirname,f'Image flattening/flat_basal_tracking')
    
    flat = np.zeros((BOTTOM_OFFSET-TOP_OFFSET,XX,XX,3))
    Iz_top = heightmap + TOP_OFFSET
    Iz_bottom = heightmap + BOTTOM_OFFSET
    for x in range(XX):
        for y in range(XX):
            
            flat[:,y,x,:] = im[Iz_top[y,x]:Iz_bottom[y,x],y,x,:]
                
    # io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    # 
        