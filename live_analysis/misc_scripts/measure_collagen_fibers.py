#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 00:06:19 2022

@author: xies
"""

from skimage import io, filters,util
import numpy as np
import pandas as pd

import scipy.ndimage as ndi

from os import path
from glob import glob
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# imfiles = glob(path.join(dirname,'Image flattening/flat_z_shift_2/t*.tif'))

XX = 460
Z = 72
T = 15

#%%

for t in range(T):
    t=9
    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131814#acks
    
    im = io.imread(path.join(dirname,f'Image flattening/flat_z_shift_2/t{t}.tif'))[...,2]
    # im = io.imread('/Users/xies/Desktop/test.tif')
    im = util.img_as_float(im)
    
    
    im_blur = filters.gaussian(im,sigma = 5)
    Gx = filters.sobel_h(im_blur)
    Gy = filters.sobel_v(im_blur)
    
    Gx = filters.gaussian(Gx,sigma=1)
    Gy = filters.gaussian(Gy,sigma=1)
    
    G = np.sqrt( Gx**2 + Gy**2)
    
    # Square the gradient for better response
    Jx = Gx**2 - Gy**2
    Jy = 2*Gx*Gy
    
    thetas = np.rad2deg(np.arctan2(Jy,Jx))
    thetas = filters.gaussian(thetas,sigma=2)
    
    # Make all angles range from 0 to 180
    thetas[thetas < 0] = thetas[thetas < 0]+180
    theta_th =thetas.copy()
    theta_th[im_blur < 0.015] = np.nan
    
    # io.imsave(path.join(dirname,f'Image flattening/collagen_fibrousness/t{t}.tif'),
    #           util.img_as_uint(G/im_blur))
    io.imsave(path.join(dirname,f'Image flattening/collagen_orientation/t{t}.tif'),
              thetas.astype(int))
    np.save(path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy'),
              theta_th)
    
#%%

plt.subplot(2,1,1);io.imshow(im_blur)
# plt.figure();io.imshow(G/im_blur)
plt.subplot(2,1,2);io.imshow(theta)

#%%

# wavelength = 10 #px
# Nthetas = 100

# t = 0

# im_blur = filters.gaussian(im,sigma = wavelength/3)

# angles = np.arange(0,np.pi,np.pi/Nthetas)
# kernels = [filters.gabor_kernel(frequency = 1./wavelength, theta = theta) for theta in angles]

# filt_imag = np.zeros([XX,XX,Nthetas])
# filt_real = np.zeros([XX,XX,Nthetas])
# for i,k in tqdm(enumerate(kernels)):
    
#     filtered = ndi.convolve(im_blur,k, mode='reflect')
#     filt_real[:,:,i] = np.real( filtered )
#     filt_imag[:,:,i] = np.imag( filtered )
    
# which = filt_real.argmax(axis=2)
# response = np.sqrt(filt_real **2 + filt_imag**2)
# for i in range(Nthetas):
#     response[:,:,i] = response[:,:,i] / response.sum(axis=2)
#     response[:,:,i] / im

# which_angle = response.argmax(axis=2)

# #%%

# alignment = angles[which_angle]
# plt.figure(); io.imshow(alignment)
    