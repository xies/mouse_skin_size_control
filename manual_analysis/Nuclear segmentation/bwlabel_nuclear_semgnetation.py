#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:41:04 2019

@author: xies
"""

from os import path
import numpy as np
from skimage import io, morphology

regions = ['/Users/xies/box/Mouse/Skin/W-R1/',
            '/Users/xies/box/Mouse/Skin/W-R2/',
            '/Users/xies/box/Mouse/Skin/W-R5/',
            '/Users/xies/box/Mouse/Skin/W-R5-full/']

for dirname in regions:
    
    selem = morphology.square(3)
    mask = io.imread(path.join(dirname,'h2b_mask.tif'))
    T,Z,X,Y = mask.shape
    
    outlines = np.zeros(mask.shape)
    for t in xrange(T):
        for z in xrange(Z):
            im = mask[t,z,...]
            mask_dilate = morphology.binary_dilation(im)
            mask_outline = mask_dilate - im
            
            outlines[t,z,...] = mask_outline
    
    io.imsave(path.join(dirname,'h2b_outlines.tif'),outlines.astype(np.int8))
    