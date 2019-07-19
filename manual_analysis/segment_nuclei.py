#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:07:30 2019

@author: xies
"""

from skimage import io,filters,morphology
import numpy as np
import pandas as pd
import os.path as path
import matplotlib.pylab as plt

dirname = '/Users/xies/Box/Mouse/Skin/W-R5/'

h2b = io.imread(path.join(dirname,'h2b_sequence.tif'))
T,Z,X,Y = h2b.shape

mask = np.zeros(h2b.shape)
for t in xrange(T):
    for z in xrange(Z):
        im = h2b[t,z,...]

        # Apply local threshold with radius = 31        
        radius = 31
        selem = morphology.disk(radius)
        localth = filters.rank.otsu(im, selem)
        
        mask[t,z,...] = (im >= localth).astype(np.int8)
        
        print 'Done with t= ',t,', z = ',z 

mask = io.imread(path.join(dirname,'h2b_mask.tif'))

# Do some clean up
for t in xrange(T):
    for z in xrange(Z):
        m = mask[t,z,...]
        m = morphology.binary_opening(m)
        m = morphology.binary_closing(m)
        mask[t,z,...] = m

io.imsave(path.join(dirname,'h2b_mask_clean.tif'),mask.astype(np.int8))

