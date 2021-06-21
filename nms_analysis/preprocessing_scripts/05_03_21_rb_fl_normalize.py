#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 13:02:36 2021

@author: xies
"""

import numpy as np
from skimage import io, util
from os import path

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/stitching'


#%%

# R = io.imread(path.join(dirname,'KO_R1.tif'))
G = io.imread(path.join(dirname,'R1_G.tif'))


#%% Normalize

# R_ = R.copy().astype(np.float64)
# for t,Rslice in enumerate(R):
#     Rslice = Rslice/Rslice.mean()
#     Rslice = Rslice - 500
#     R_[t,:,:,:] = Rslice
    
G_ = G.copy().astype(np.float64)
for t,Gslice in enumerate(G):
    
    Gslice = Gslice/Gslice.mean()
    Gslice = Gslice
    G_[t,:,:,:] = Gslice
    
    

#%%

# io.imsave(path.join(dirname,'KO_R1_Rc.tif'),util.img_as_uint(R_/R_.max()))
io.imsave(path.join(dirname,'R1_Gc.tif'),util.img_as_uint(G_/G_.max()))

