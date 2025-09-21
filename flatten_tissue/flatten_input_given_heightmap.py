#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:58:23 2022

@author: xies
"""

import numpy as np
from skimage import io
from glob import glob
from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-168hr_R2.tif'))

#%% Load a heightmap and flatten the given z-stack

OFFSET = 2


imstack = io.imread(filenames[0])
T,Z,XX,_,C = imstack.shape

# im2flatten = imstack[...,1] # flatten green channel
im2flatten = imstack

for t,im in tqdm(enumerate(im2flatten)):
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    output_dir = path.join(dirname,f'Image flattening/flat_z_shift_{OFFSET}')
    
    flat = np.zeros((XX,XX,3))
    Iz = heightmap + OFFSET
    for x in range(XX):
        for y in range(XX):
            flat[y,x,:] = im[Iz[y,x],y,x,:]
                
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    
        