#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:17:20 2022

@author: xies

Optimized for Mesa et al organization
"""

import numpy as np
from skimage import io, filters, exposure, util
from os import path
from glob import glob

from tqdm import tqdm
from scipy.optimize import curve_fit

#%%

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
filenames = glob(path.join(dirname,'Cropped_images/20161127_Fucci_1F_0-168hr_W_R1_cropped.tif'))
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Confocal/08-26-2022/10month 2week induce/Paw H2B-CFP FUCCI2 Phall647/WT1'
# filenames = glob(path.join(dirname,'WT1.tif'))

channel2use = 1

def logit_curve(x,L,k,x0):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

imstack = io.imread(filenames[0])

XX = 460
ZZ = 72

#%%

XY_sigma = 25
Z_sigma = 5

TOP_Z_BOUND = 35
BOTTOM_Z_BOUND = 65

z_shift = 10

# im_list = map(lambda f: io.imread(f)[channel2use,...], filenames)

# for t,im in tqdm(enumerate(im_list)):
# for t,im in tqdm(enumerate(imstack)):

t = 1
im = imstack[t,...]
        
im_xy_blur = np.zeros_like(im[:,:,:,channel2use],dtype=float)

#XY_blur
for z,im_ in enumerate(im[:,:,:,channel2use]):
    im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma)
    

#Z_blur
im_z_blur = np.zeros_like(im_xy_blur)
for x in tqdm(range(XX)):
    for y in range(XX):
        im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)

io.imsave(path.join(dirname,'xyz_blur_'+'t0.tif'), util.img_as_int(im_z_blur))

# Derivative of R_sgh wrt Z -> Take the max dI/dz for each (x,y) position
_tmp = im_z_blur.copy()
_tmp[np.isnan(_tmp)] = 0
heightmap = _tmp.argmax(axis=0)
heightmap = np.diff(_tmp[TOP_Z_BOUND:BOTTOM_Z_BOUND,...],axis=0).argmax(axis=0) + TOP_Z_BOUND


io.imsave(path.join(dirname,f'Image flattening/heightmaps/t{1}.tif'),heightmap.astype(np.int8))

# Reconstruct flattened movie


Iz = np.round(heightmap + z_shift).astype(int)

# NB: tried using np,take and np.choose, doesn't work bc of size limit. DO NOT use np.take
flat = np.zeros((XX,XX,3))
height_image = np.zeros_like(im)
for x in range(XX):
    for y in range(XX):
        flat[y,x,:] = im[Iz[y,x],y,x,:]
        height_image[Iz[y,x],y,x] = 1

io.imsave(path.join(dirname,f'Image flattening/flat_z_shift_{z_shift}/t{t+1}.tif'), flat.astype(np.int16))
io.imsave(path.join(dirname,f'Image flattening/height_image/t{t}.tif'), height_image.astype(np.int16))

pd.Series({'XY_sigma':XY_sigma,'Z_sigma':Z_sigma,TOP_Z_BOUND:'TOP_Z_BOUND','BOTTOM_Z_BOUND':BOTTOM_Z_BOUND,
              'z_shift':z_shift}).to_csv(path.join(dirname,f'Image flattening/params/t{t}.csv'))

