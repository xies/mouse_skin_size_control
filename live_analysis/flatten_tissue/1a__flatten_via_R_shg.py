#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:32:00 2022

@author: xies
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, util
from os import path
from glob import glob
from re import match

from tqdm import tqdm
from scipy.optimize import curve_fit

from ifUtils import min_normalize_image
from twophoton_util import parse_aligned_timecourse_directory

#%%

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1/'

filelist = parse_aligned_timecourse_directory(dirname)

XX = 1024
ZZ = 95
T = 19
channel2use = 'R_shg'

def logit_curve(x,L,k,x0):
    # I = ~np.isnan(
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

#%% Calculate heightmaps
'''
1. XY-blur and then Z-blur with generous kernel sizes on R_shg stack
2. For all (x,y) pairs take the derivative of the image
3. Take the arg-maximum of the derivative (within a predefined range) as the Iz stack
'''

OVERWRITE = False

XY_sigma = 25
Z_sigma = 2.5

TOP_Z_BOUND = 45
BOTTOM_Z_BOUND = 75

OFF_SET = 0

for t in tqdm(range(T)):
    
    f = filelist[channel2use].iloc[t]
    out_dir = path.split(path.dirname(f))[0]
    if path.exists(path.join(dirname,f'flat/t{t}.tif')) and not OVERWRITE:
        continue
    
    
    im = io.imread(f).astype(float)
    im = (im /im.max())* (2**16 -1)
    
    im_xy_blur = np.zeros_like(im,dtype=float)
    
    #XY_blur
    for z,im_ in enumerate(im):
        im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma)
    
    #Z_blur
    im_z_blur = np.zeros_like(im_xy_blur)
    im[:,np.all(im == 0,axis=0)] = np.nan
    im[np.all(np.all(im == 0,axis=1),axis=1),...] = np.nan
    for x in tqdm(range(XX)):
        for y in range(XX):
            im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)
    
    # io.imsave(path.join(dirname,'blur.tif'), util.img_as_int(im_z_blur/im_z_blur.max()))
    
    # Derivative of R_sgh wrt Z -> Take the max dI/dz for each (x,y) position
    _tmp = im_z_blur.copy()
    _tmp[np.isnan(_tmp)] = 0
    heightmap = _tmp.argmax(axis=0)
    heightmap = np.diff(_tmp[TOP_Z_BOUND:BOTTOM_Z_BOUND,...],axis=0).argmax(axis=0) + TOP_Z_BOUND
    
    io.imsave(path.join(out_dir,'heightmap.tif'),heightmap.astype(np.int8))
        



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
    
    
    
    

    