#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:17:20 2022

@author: xies

Optimized for Mesa et al organization
"""

import numpy as np
import pandas as pd
from skimage import io, filters, exposure, util
from os import path
from glob import glob

from tqdm import tqdm
from scipy.optimize import curve_fit

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-*.tif'))
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Confocal/02-11-2023 Rb Cre-plusminus Tamoxifen control/H2B Cerulean FUCCI2 Phall-647/WT1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/11-17-2022 RB-KO tam control/M9 RB noCre tam/R1/'
# filenames = glob(path.join(dirname,'tracking/H2B_stack.tif'))
filenames = glob(path.join(dirname,'*.tif'))
                 

channel2use = 1

def logit_curve(x,L,k,x0):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

imstack = io.imread(filenames[0])

XX = 1024
ZZ = 84

#%%

XY_sigma = 25
Z_sigma = 10

TOP_Z_BOUND = 3
BOTTOM_Z_BOUND = 25

z_shift = 2


OVERWRITE = True
# im_list = map(lambda f: io.imread(f)[channel2use,...], filenames)

# for t,im in tqdm(enumerate(im_list)):
# for t,im in tqdm(enumerate(imstack)):

# t = 0
    # im = imstack[t,...]
im = imstack

# if path.exists(path.join(dirname,f'Image flattening/params/t{t}.csv')) and not OVERWRITE:
#     params = pd.read_csv(path.join(dirname,f'Image flattening/params/t{t}.csv'),index_col=0,header=0).T
#     XY_sigma = params['XY_sigma']
#     Z_sigma = params['Z_sigma']
#     TOP_Z_BOUND = params['TOP_Z_BOUND']
#     BOTTOM_Z_BOUND = params['BOTTOM_Z_BOUND']
    
im_xy_blur = np.zeros_like(im[:,:,:],dtype=float)

#XY_blur
for z,im_ in enumerate(im[:,:,:]):
    im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma)
    
#Z_blur
im_z_blur = np.zeros_like(im_xy_blur)
for x in range(XX):
    for y in range(XX):
        im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)
        
# io.imsave(path.join(dirname,f'Image flattening/xyz_blur/t{t}.tif'), util.img_as_int(im_z_blur),check_contrast=False)
io.imsave(path.join(dirname,f'R1_xyz_blur.tif'), util.img_as_int(im_z_blur),check_contrast=False)


# Derivative of R_sgh wrt Z -> Take the max dI/dz for each (x,y) position
_tmp = im_z_blur.copy()
_tmp[np.isnan(_tmp)] = 0
heightmap = np.diff(_tmp,axis=0).argmax(axis=0)
heightmap[heightmap > BOTTOM_Z_BOUND] = BOTTOM_Z_BOUND
heightmap[heightmap < TOP_Z_BOUND] = TOP_Z_BOUND

# io.imsave(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'), heightmap.astype(np.int16),check_contrast=False)
io.imsave(path.join(dirname,f'R1_height_map.tif'), heightmap.astype(np.int16),check_contrast=False)

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
# io.imsave(path.join(dirname,f'Image flattening/height_image/t{t}.tif'), height_image.astype(np.int16),check_contrast=False)
io.imsave(path.join(dirname,f'R1_height_img.tif'), height_image.astype(np.int16),check_contrast=False)

    
# pd.Series({'XY_sigma':XY_sigma,'Z_sigma':Z_sigma,'TOP_Z_BOUND':TOP_Z_BOUND,'BOTTOM_Z_BOUND':BOTTOM_Z_BOUND,
#               'z_shift':z_shift}).to_csv(path.join(dirname,f'Image flattening/params/t{t}.csv'))

