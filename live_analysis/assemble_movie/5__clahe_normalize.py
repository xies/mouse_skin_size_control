#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:05:14 2023

@author: xies
"""

import numpy as np
from skimage import io,exposure, filters, util
from os import path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Right ear/Post Ethanol/R3'

#%% Locally histogram-normalize

im = io.imread(path.join(dirname,'master_stack/G.tif'))

kernel_size = (im.shape[1] // 3, #~25
               im.shape[2] // 4, #~128
               im.shape[3] // 4)
kernel_size = np.array(kernel_size)

im_clahe = np.zeros_like(im,dtype=float)


# clahe_blur = np.zeros_like(im,dtype=float)
for t, im_time in tqdm(enumerate(im)):
    im_clahe[t,...] = exposure.equalize_adapthist(im_time/im_time.max(), kernel_size=kernel_size, clip_limit=0.01, nbins=256)
    # clahe_blur[t,...] = gaussian_filter(im_clahe[t,...],sigma=[.5,.5,.5])
io.imsave(path.join(dirname,'master_stack/G_clahe.tif'),util.img_as_uint(im_clahe))

# 3d Blur
# io.imsave(path.join(dirname,'master_stack/B_clahe_blur.tif'),util.img_as_uint(clahe_blur))

