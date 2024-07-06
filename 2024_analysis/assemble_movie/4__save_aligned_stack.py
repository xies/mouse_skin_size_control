#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:53:19 2022

@author: xies
"""


import numpy as np
import pandas as pd
from skimage import io, util
from os import path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from twophotonUtils import parse_aligned_timecourse_directory

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Old mice/01-24-2024 12month old mice/F1 DOB 12-18-2022/R1'

filelist = parse_aligned_timecourse_directory(dirname,folder_str='*.*',SPECIAL=[1])

#%% Stack channels & save into single tif

G = []
G_blur = []
B = []
B_blur = []
R = []
R_shg = []

for t in tqdm(filelist.index):
    
    G_ = io.imread(filelist.loc[t,'G'])
    G.append(G_)
    G_blur.append(gaussian_filter(G_,sigma=[.5,.5,.5]))
    
    R_ = io.imread(filelist.loc[t,'R'])
    R.append(R_)
    
    R_shg_ = io.imread(filelist.loc[t,'R_shg'])
    R_shg.append(R_shg_)
    
    B_ = io.imread(filelist.loc[t,'B'])
    B.append(B_)
    B_blur.append(gaussian_filter(B_,sigma=[.5,.5,.5]))
    
print('Saving G ...')
io.imsave(path.join(dirname,'master_stack/G.tif'), np.stack(G).astype(np.uint16))
print('Saving R ...')
io.imsave(path.join(dirname,'master_stack/R.tif'), np.stack(R).astype(np.uint16))
print('Saving R_shg ...')
io.imsave(path.join(dirname,'master_stack/R_shg.tif'), np.stack(R_shg).astype(np.uint16))
print('Saving B ...')
io.imsave(path.join(dirname,'master_stack/B.tif'), np.stack(B).astype(np.uint16))

# print('Saving G_blur ...')
# io.imsave(path.join(dirname,'master_stack/G_blur.tif'),np.stack(G_blur).astype(np.uint16))
# print('Saving B_blur ...')
# io.imsave(path.join(dirname,'master_stack/B_blur.tif'),np.stack(B_blur).astype(np.uint16))




