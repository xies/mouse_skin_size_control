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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-168hr_W_R1_cropped.tif'))

filenames = glob(path.join(dirname,'manual_basal_tracking/basal_tracks.tif'))

#%% Load a heightmap and flatten the given z-stack

OFFSET = 2
Zoffsetrange = np.arange(-20,5,1).astype(int)


imstack = io.imread(filenames[0])
T,Z,XX,_ = imstack.shape

im2flatten = imstack # flatten green channel

Z = len(Zoffsetrange)


for t,im in tqdm(enumerate(im2flatten)):
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    # output_dir = path.join(dirname,f'Image flattening/flat_z_shift_{OFFSET}')
    output_dir = path.join(dirname,'Image flattening/flat_basal_tracking/')
    
    flat = np.zeros((Z,XX,XX))
    for z,OFFSET in enumerate(Zoffsetrange):
        Iz = heightmap + OFFSET
        for x in range(XX):
            for y in range(XX):
                flat[z,y,x] = im[Iz[y,x],y,x]
                
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    
        