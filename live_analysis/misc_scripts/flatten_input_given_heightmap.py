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

#%% Load a heightmap and flatten the given z-stack

OFFSET = 10

imstack = io.imread(filenames[0])

im2flatten = imstack # flatten green channel
T,Z,XX,_,C = imstack.shape


for t,im in tqdm(enumerate(im2flatten)):
    Iz = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    Iz += OFFSET
    output_dir = path.join(dirname,'Image flattening/flat_G')
    
    flat = np.zeros((XX,XX,C))
    for x in range(XX):
        for y in range(XX):
            flat[y,x,:] = im[Iz[y,x],y,x,:]
            
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    
    