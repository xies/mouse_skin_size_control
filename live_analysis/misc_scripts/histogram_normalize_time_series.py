#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 16:10:30 2025

@author: xies
"""

from skimage import util, io, exposure
from os import path
import numpy as np

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

B = io.imread(path.join(dirname,'Cropped_images/B.tif'))
B_normed = np.zeros_like(B,dtype=float)

for t,im in enumerate(B):
    B_normed[t,...] = exposure.equalize_hist(im,nbins=2**16-1)

io.imsave(path.join(dirname,'Cropped_images/histogram_normalized/B.tif'),
          util.img_as_uint(B_normed))


R = io.imread(path.join(dirname,'Cropped_images/R.tif'))
R_normed = np.zeros_like(R,dtype=float)

for t,im in enumerate(R):
    R_normed[t,...] = exposure.equalize_hist(im,nbins=2**16-1)
io.imsave(path.join(dirname,'Cropped_images/histogram_normalized/R.tif'),
          util.img_as_uint(R_normed))

G = io.imread(path.join(dirname,'Cropped_images/G.tif'))
G_normed = np.zeros_like(G,dtype=float)
for t,im in enumerate(G):
    G_normed[t,...] = exposure.equalize_hist(im,nbins=2**16-1)

io.imsave(path.join(dirname,'Cropped_images/histogram_normalized/G.tif'),
          util.img_as_uint(G_normed))

