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


# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R2/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-14-2023 R26CreER Rb-fl no tam ablation 24hr/M5 white/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R1'

filelist = parse_aligned_timecourse_directory(dirname,folder_str='*.*',INCLUDE_ZERO=False)

#%% Stack channels & save into single tif

G = []
G_blur = []
B = []
B_blur = []
R = []
R_shg = []

for t in tqdm(range(len(filelist))):
    
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

print('Saving G_blur ...')
io.imsave(path.join(dirname,'master_stack/G_blur.tif'),np.stack(G_blur).astype(np.uint16))
print('Saving B_blur ...')
io.imsave(path.join(dirname,'master_stack/B_blur.tif'),np.stack(B_blur).astype(np.uint16))




