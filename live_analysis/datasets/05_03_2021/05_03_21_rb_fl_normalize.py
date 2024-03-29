#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:02:36 2021

@author: xies
"""

import numpy as np
from skimage import io, util
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M2 RB-KO/R1/stitched'


#%%

R = io.imread(path.join(dirname,'current_r.tif'))
G = io.imread(path.join(dirname,'current_g.tif'))
B = io.imread(path.join(dirname,'current_b.tif'))



R_ = R.copy().astype(np.float64)
for t,Rslice in enumerate(R):
    Rslice = Rslice/Rslice.mean()
    Rslice = Rslice
    R_[t,:,:,:] = Rslice
    
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

#%%

io.imsave(path.join(dirname,'Rc.tif'),util.img_as_uint(R_/R_.max()))
io.imsave(path.join(dirname,'Gc.tif'),util.img_as_uint(G_/G_.max()))
io.imsave(path.join(dirname,'Bc.tif'),util.img_as_uint(B_/B_.max()))


