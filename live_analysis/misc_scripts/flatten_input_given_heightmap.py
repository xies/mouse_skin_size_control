#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:58:23 2022

@author: xies
"""

import numpy as np
from skimage import io

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-168hr_W_R1_cropped.tif'))

#%% Load a heightmap and flatten the given z-stack

imstack = io.imread(filenames[0])

im2flatten = imstack[...,1] # flatten green channel


for t,im in enumerate(im2flatten):
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    output_dir = path.join(dirname,'Image flattening/flat_G')
    flat = im[heightmap,:,:]
    io.imsave( path.join(output_dir,f't{t}.tif'))
    