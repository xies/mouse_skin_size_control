#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 17:05:14 2023

@author: xies
"""

import numpy as np
from skimage import io,exposure, filters, util
from os import path
from tqdm import tqdm

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R2'

# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'

# G = io.imread(path.join(dirname,'master_stack/G.tif'))
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/Cropped_images/'


#%% Locally histogram-normalize

im = io.imread(path.join(dirname,'master_stack/G.tif'))

kernel_size = (im.shape[1] // 3, #~25
               im.shape[2] // 4, #~128
               im.shape[3] // 4)
kernel_size = np.array(kernel_size)

im_clahe = np.zeros_like(im,dtype=float)
for t, im_time in tqdm(enumerate(im)):
    im_clahe[t,...] = exposure.equalize_adapthist(im_time, kernel_size=kernel_size, clip_limit=0.01, nbins=256)

io.imsave(path.join(dirname,'master_stack/G_clahe.tif'),util.img_as_uint(im_clahe))

