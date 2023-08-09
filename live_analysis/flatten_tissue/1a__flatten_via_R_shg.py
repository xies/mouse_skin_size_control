#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:32:00 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, util
from os import path
from scipy.ndimage import gaussian_filter

from tqdm import tqdm

from ifUtils import min_normalize_image
from twophotonUtils import parse_aligned_timecourse_directory

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2/'


filelist = parse_aligned_timecourse_directory(dirname)

XX = 1024
ZZ = 95
T = 16
channel2use = 'R_shg'

#%% Calculate heightmaps
'''
1. XY-blur and then Z-blur with generous kernel sizes on R_shg stack
2. For all (x,y) pairs take the derivative of the image
3. Take the arg-maximum of the derivative (within a predefined range) as the Iz stack
'''

OVERWRITE = True

XY_sigma = 30
Z_sigma = 10

TOP_Z_BOUND = 40
BOTTOM_Z_BOUND = 5

OFF_SET = 0

z_shift = 0

for t in tqdm(range(T)):
    
    f = filelist[channel2use].iloc[t]
    out_dir = path.split(path.dirname(f))[0]
    if path.exists(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif')) and not OVERWRITE:
        continue
    
    
    im = io.imread(f).astype(float)
    im = (im /im.max())* (2**16 -1)
    
    # im_xy_blur = np.zeros_like(im,dtype=float)
    im_xyz_blur = gaussian_filter(im,sigma = [Z_sigma,XY_sigma,XY_sigma])
    
    # #XY_blur
    # for z,im_ in enumerate(im):
    #     im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma)
    
    # #Z_blur
    # im_z_blur = np.zeros_like(im_xy_blur)
    # im[:,np.all(im == 0,axis=0)] = np.nan
    # im[np.all(np.all(im == 0,axis=1),axis=1),...] = np.nan
    # for x in range(XX):
    #     for y in range(XX):
    #         im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)
    io.imsave(path.join(dirname,f'Image flattening/xyz_blur/t{t}.tif'), im_xyz_blur.astype(np.int16),check_contrast=False)
    
    # Derivative of R_sgh wrt Z -> Take the max dI/dz for each (x,y) position
    _tmp = im_xyz_blur.copy()
    _tmp[np.isnan(_tmp)] = 0
    heightmap = np.diff(-_tmp[BOTTOM_Z_BOUND:TOP_Z_BOUND,...],axis=0).argmax(axis=0)
    heightmap = heightmap + BOTTOM_Z_BOUND
    
    io.imsave(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'), heightmap.astype(np.int16),check_contrast=False)
    # io.imsave(path.join(dirname,f'R1_height_map.tif'), heightmap.astype(np.int16),check_contrast=False)
    
    # Reconstruct flattened movie
    Iz = np.round(heightmap + z_shift).astype(int)
    
    # NB: tried using np,take and np.choose, doesn't work bc of size limit. DO NOT use np.take
    flat = np.zeros((XX,XX))
    height_image = np.zeros_like(im)
    for x in range(XX):
        for y in range(XX):
            flat[y,x] = im[Iz[y,x],y,x]
            height_image[Iz[y,x],y,x] = 1
    
    io.imsave(path.join(dirname,f'Image flattening/height_image/t{t}.tif'), height_image.astype(np.int16),check_contrast=False)


#%% Reconstruct flattened movie and return a boundary image

OVERWRITE = True
DEBUG = True
OFF_SET = -5

# Reload filelist
filelist = parse_timecourse_directory(dirname)

assert('Heightmap' in filelist.columns)
    
for t in tqdm(range(T)):
    if path.exists(path.join(dirname,f'flat/t{t}.tif')) and not OVERWRITE:
        continue
    
    heightmap = io.imread(filelist['Heightmap'].iloc[t])
    
    # Reconstruct flattened movie
    Iz = np.round(heightmap + OFF_SET).astype(int)
    
    G_im = io.imread(filelist['G'].iloc[t]).astype(float)
    G_im = min_normalize_image(G_im)
    
    # NB: tried using np.take and np.choose, doesn't work bc of size limit. DO NOT use np.take
    flat = np.zeros((XX,XX))
    height_image = np.zeros_like(im)
    for x in range(XX):
        for y in range(XX):
            flat[y,x] = G_im[Iz[y,x],y,x]
            if DEBUG:
                height_image[Iz[y,x],y,x] = 1
    
    io.imsave(path.join(dirname,f'flat/t{t}.tif'), flat.astype(np.uint16))
    
    
    
    

    