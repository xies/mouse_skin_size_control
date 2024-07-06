#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:49:45 2022

@author: xies
"""

import numpy as np
from skimage import io
from glob import glob
from os import path


#%%

# dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1/reg'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M6 RBKO/R1/'

imgfiles = glob(path.join(dirname,'master_stack/G.tif'))

im = io.imread(imgfiles[0])

# for f in imgfiles:
#     im = io.imread(f)
#     channel = im[1,...]
    
#     path.splitext(f)
    
#     io.imsave(path.dirname(f)[0]+'_chan1.tif',channel)
    
#%%

for t,im_t in tqdm(enumerate(im)):
    io.imsave( path.join( dirname, 'im_seq', f't{t}.tif'), im_t[...])
     
    