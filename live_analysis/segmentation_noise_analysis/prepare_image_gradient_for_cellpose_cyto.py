#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:53:11 2023

@author: xies
"""

import numpy as np
from skimage import io, util
from scipy import ndimage

from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'


B = io.imread(path.join(dirname,'master_stack/B_blur.tif'))

#%%

edges = np.zeros_like(B,dtype=float)
for i,im in tqdm(enumerate(B)):
    Gz = ndimage.sobel(im.astype(float),axis=0)
    Gy = ndimage.sobel(im.astype(float),axis=1)
    Gx = ndimage.sobel(im.astype(float),axis=2)
    edges[i,...] = np.sqrt(Gz**2 + Gy**2 + Gx**2)


io.imsave(path.join(dirname,'master_stack/B_blur_sobel.tif'),util.img_as_uint(edges/edges.max()))

