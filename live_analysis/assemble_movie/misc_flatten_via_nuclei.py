#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 22:17:20 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, exposure, util
from os import path
from glob import glob

from tqdm import tqdm

#%%

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3/reg'
filenames = glob(path.join(dirname,'reg*.tif'))

im_list = list(map(io.imread,filenames))

XX = 1024
ZZ = 44


from scipy.optimize import curve_fit

def logit_curve(x,L,k,x0):
    y = L / (1 + np.exp(-k*(x-x0)))
    return y

#%% XY Gaussian blur + Z blur


XY_sigma = 35
Z_sigma = 5

for t,im in enumerate(im_list):

    
    im_xy_blur = np.zeros_like(im[...,0],dtype=float)
    
    #XY_blur
    for z,im_ in enumerate(im[...,0]):
        im_xy_blur[z,...] = filters.gaussian(im_,sigma = XY_sigma)
        
    
    
    
    #Z_blur
    im_z_blur = np.zeros_like(im_xy_blur)
    for x in tqdm(range(XX)):
        for y in range(XX):
            im_z_blur[:,y,x] = filters.gaussian(im_xy_blur[:,y,x], sigma= Z_sigma)
    
    io.imsave(path.join(dirname,f'XYZ_blurred/t{t+1}.tif'), util.img_as_int(im_xy_blur))

    k = np.zeros((XX,XX))
    x0 = np.zeros((XX,XX))
    
    for x in tqdm(np.arange(XX)):
        for y in np.arange(XX):
    
            Y = heightmap[:,y,x]
            if Y.max() < 20:
                print('Skip')
                continue
            
            X = np.arange(44)
            
            p,_ = curve_fit(logit_curve,X,Y,p0=[Y.max(),1,10])
            
            k[y,x] = p[1]
            x0[y,x] = p[2]
        
    half = x0 / k
    half[np.isnan(half)] = 0
    
    io.imsave(path.join(dirname,f'heightmaps/t{t+1}.tif'),half.astype(np.int8))

    # Reconstruct flattened movie
    z_shift = 25
    
    Iz = np.round(half + z_shift).astype(int)
    
    # NB: tried using np,take and np.choose, doesn't work bc of size limit. DO NOT use np.take
    flat = np.zeros((XX,XX,3))
    for x in range(XX):
        for y in range(XX):
            flat[y,x,:] = im[Iz[y,x],y,x,:]

    io.imsave(path.join(dirname,f'flat/t{t+1}.tif'), flat.astype(np.int16))





