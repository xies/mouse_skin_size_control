#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:30:55 2023

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io,util
from os import path
from glob import glob

dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Pre tam/R1/1. Day 0.5'
raw = io.imread(path.join(dirname,'B_reg.tif'))
warped = io.imread(path.join(dirname,'B_warp.tif'))
dfield = io.imread(path.join(dirname,'dfield.tif'))

#%%

# test_image = np.zeros_like(raw)

# # test_image[40,500,500] = (2**16)-1
# test_image[40,100,100] = (2**16)-1
# # test_image[40,800,800] = (2**16)-1

# io.imsave('/Users/xies/Desktop/test.tif',test_image)

# #%%

# test_warped = io.imread('/Users/xies/Desktop/test_warp.tif')

# #%% Read out the 'rough' input coords

# input_coords = np.array([40,100,100])
# warped_coords = (input_coords - dfield[40,100,100]).astype(int)

# target_coords = np.array(list(map(np.mean,np.where(test_warped > 0))))

# actual_diff = target_coords - input_coords

# print(f'Actual diff {actual_diff}')

#%%

pts = pd.read_csv('/Users/xies/Desktop/pts.csv')

coords2unwarp = np.round(pts[['axis-0','axis-1','axis-2']].values.astype(int))

new_coords = [c + dfield[tuple(c)][::-1] for c in coords2unwarp]

new_pts = pd.DataFrame(new_coords,columns=['axis-0','axis-1','axis-2'])
new_pts.to_csv('/Users/xies/Desktop/new_pts.csv')

