#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:17:20 2022

@author: xies

Optimized for Mesa et al organization
"""

import numpy as np
import pandas as pd
from skimage import io
from os import path

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2/'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R2'

XX = 1024
ZZ = 95

imstack = io.imread(path.join(dirname,'master_stack/R_shg.tif'))
ZZ = imstack.shape[1]
TT = imstack.shape[0]

SIGN = -1

#%%

XY_sigma = 25
Z_sigma = 10

TOP_Z_BOUND = 10
BOTTOM_Z_BOUND = 40

z_shift = 0

OVERWRITE = True

for t in tqdm(range(imstack.shape[0])):

    # im = io.imread(filelist.loc[t,channel2use])
    im = imstack[t,...]
    
    if path.exists(path.join(dirname,f'Image flattening/params/t{t}.csv')) and not OVERWRITE:
        params = pd.read_csv(path.join(dirname,f'Image flattening/params/t{t}.csv'),index_col=0,header=0).T
        XY_sigma = params['XY_sigma'].values
        Z_sigma = params['Z_sigma'].values
        TOP_Z_BOUND = params['TOP_Z_BOUND'].values
        BOTTOM_Z_BOUND = params['BOTTOM_Z_BOUND'].values
        continue
    
    im_z_blur = gaussian_filter(im,sigma=[Z_sigma,XY_sigma,XY_sigma])
    # im_xy_blur = np.zeros_like(im[:,:,:],dtype=float)
    #XY_blur
    # for z,im_ in enumerate(im[:,:,:]):
    #     im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma) 
    
        
    # #Z_blur
    # im_z_blur = np.zeros_like(im_xy_blur)
    # for x in range(XX):
    #     for y in range(XX):
    #         im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)
            
    # io.imsave(path.join(dirname,f'Image flattening/xyz_blur/t{t}.tif'), util.img_as_int(im_z_blur),check_contrast=False)
    
    
    _tmp = im_z_blur.copy().astype(float)
    # Derivative of R_sgh wrt Z -> Take the max dI/dz for each (x,y) position
    _tmp[np.isnan(_tmp)] = 0
    _tmp_diff = SIGN * np.diff(_tmp,axis=0)
    heightmap = _tmp_diff.argmax(axis=0)
    heightmap[heightmap > BOTTOM_Z_BOUND] = BOTTOM_Z_BOUND
    heightmap[heightmap < TOP_Z_BOUND] = TOP_Z_BOUND
    
    io.imsave(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'), heightmap.astype(np.uint16),check_contrast=False)
    
    # Reconstruct flattened movie
    Iz = np.round(heightmap + z_shift).astype(int)
    
    # NB: tried using np,take and np.choose, doesn't work bc of size limit. DO NOT use np.take
    flat = np.zeros((XX,XX))
    height_image = np.zeros_like(im)
    for x in range(XX):
        for y in range(XX):
            flat[y,x] = im[Iz[y,x],y,x]
            height_image[Iz[y,x],y,x] = 1
    
    # io.imsave(path.join(dirname,f'Image flattening/flat_z_shift_{z_shift}/t{t}.tif'), flat.astype(np.int16),check_contrast=False)
    io.imsave(path.join(dirname,f'Image flattening/height_image/t{t}.tif'), height_image.astype(np.int16),check_contrast=False)
    # io.imsave(path.join(dirname,f'R1_height_img.tif'), height_image.astype(np.int16),check_contrast=False)

    
    pd.Series({'XY_sigma':XY_sigma,'Z_sigma':Z_sigma,'TOP_Z_BOUND':TOP_Z_BOUND,'BOTTOM_Z_BOUND':BOTTOM_Z_BOUND,
              'z_shift':z_shift}).to_csv(path.join(dirname,f'Image flattening/params/t{t}.csv'))

