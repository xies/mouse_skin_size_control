#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:59:35 2021

@author: xies
"""

import numpy as np
from skimage import io, util
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/07-15-2021/WT Breeder/stitching'

#%%

# R = io.imread(path.join(dirname,'R.tif'))
G = io.imread(path.join(dirname,'G.tif'))
B = io.imread(path.join(dirname,'B.tif'))

#%% Normalize

# R_ = R.copy().astype(np.float64)
# for t,Rslice in enumerate(R):
#     Rslice = Rslice/Rslice.mean()
#     Rslice = Rslice
#     R_[t,:,:,:] = Rslice
    
G_ = G.copy().astype(np.float64)
for t,Gslice in enumerate(G):

    Gslice = Gslice/Gslice.mean()
    Gslice = Gslice
    G_[t,:,:,:] = Gslice

B_ = B.copy().astype(np.float64)
for t,Bslice in enumerate(B):
    
    Bslice = Bslice/Bslice.mean()
    Bslice = Bslice
    B_[t,:,:,:] = Bslice
        

#%% Save images

# io.imsave(path.join(dirname,'Rc.tif'),util.img_as_uint(R_/R_.max()))
io.imsave(path.join(dirname,'Gc.tif'),util.img_as_uint(G_/G_.max()))
io.imsave(path.join(dirname,'Bc.tif'),util.img_as_uint(B_/B_.max()))


