#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 00:06:19 2022

@author: xies

NB: Probably more useful to calculate
    
"""

from skimage import io, filters,util
import numpy as np
import pandas as pd

import scipy.ndimage as ndi

from os import path
from glob import glob
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
# imfiles = glob(path.join(dirname,'Image flattening/flat_z_shift_2/t*.tif'))

XX = 460
Z = 70
T = 1

#%%

for t in range(T):
    # t=0
    
    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0131814#acks
    
    im = io.imread(path.join(dirname,f'Image flattening/flat_z_shift_2/t{t}.tif'))[...,2]
    # im = io.imread('/Users/xies/Desktop/test.tif')
    im = util.img_as_float(im)
    
    
    im_blur = filters.gaussian(im,sigma = 3)
    Gx = filters.sobel_h(im_blur)
    Gy = filters.sobel_v(im_blur)
    
    
    # io.imsave(path.join(dirname,f'Image flattening/collagen_fibrousness/t{t}.tif'),
    #           util.img_as_uint(G/im_blur))
    # io.imsave(path.join(dirname,f'Image flattening/collagen_orientation/t{t}.tif'),
    #           thetas.astype(int))
    # np.save(path.join(dirname,f'Image flattening/collagen_orientation/t{t}.npy'),
    #           [Gx,Gy])
    
#%%

plt.subplot(2,1,1);io.imshow(im_blur)
# plt.figure();io.imshow(G/im_blur)
plt.subplot(2,1,2);io.imshow(theta)

    