#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:46:11 2021

Based on: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html#sphx-glr-auto-examples-color-exposure-plot-adapt-hist-eq-3d-py

@author: xies
"""

import numpy as np
from skimage import io, util
from skimage.exposure import equalize_adapthist
from os import path
from glob import glob

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M2 RB-KO/R1/'

#%%

def nonans(x):
    return x[~np.isnan(x)]

flist = glob(path.join(dirname,'*/*_c.tif'))

#%% Normalize (adaptive hist / CLAHE) 3D

for f in flist:
    im = io.imread(f)[:,:,:,1].astype(float)
    im = im.transpose() # Reorder to XYZ
    [X,Y,Z] = im.shape
    
    im[im==0] = np.nan # Trim off 0s, which are introduced mostly as part of stitching
    
    im_clip = (im - np.nanmin(im)) / (np.nanmax(im) - np.nanmin(im))
    # Put back the zeros
    im_clip[ np.isnan(im_clip) ] = 0
    
    # Initiate kernel
    kernel = np.array( [ X // 4, Y // 4, Z // 1 ] )
     
    equalized = equalize_adapthist(im_clip, kernel_size=kernel, clip_limit = 1)
    equalized = equalized.transpose()
    
    equalized = equalized-equalized.min()
    
    io.imsave(path.splitext(f)[0] + '_eq.tif',util.img_as_uint(equalized))
    print(f'Saved {f}')

# #%% Normalize (2D)

# for f in flist:
#     im = io.imread(f).astype(float)
#     print(f'Loaded {f}')
#     # Convert all 0 into NaN
#     im[ im == 0] = np.nan
    
#     equalized = np.zeros_like(im)
    
#     for c in range(im.shape[3]):
        
#         this_channel = im[...,c]
#         # Rectify to non-NaN zero
#         min_int = np.nanmin(this_channel)
#         this_channel[ np.isnan(this_channel) ] = 0
#         this_channel -= min_int
#         this_channel[ this_channel < 0 ] = 0
        
#         # Run CLAHE slice-wize
#         for z in range(im.shape[0]):
#             equalized[z,:,:,c] = equalize_adapthist(this_channel[z,...].astype(np.int16),256)
            
#     print(f'Equalized {f}')

#     # save
#     io.imsave(path.splitext(f)[0] + '_eq.tif', util.img_as_uint(equalized))
#     print(f'Saved {f}')

