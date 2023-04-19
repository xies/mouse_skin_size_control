#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure, exposure, util, segmentation
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

dirnames = {}
# dirnames['WT R2'] = '/Users/xies//OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M1 WT/R1/'
# dirnames['WT R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'

dirnames['RBKO R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKOs/R2'

dx = 0.2920097/1.5
# dx = 1

RECALCULATE = True

dist2expand = 3

#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)

for name,dirname in dirnames.items():
    
    print(f'Working on {name}')
    
    genotype = name.split(' ')[0]
    
    # filtered_segs = io.imread(path.join(dirname,'manual_tracking/filtered_segmentation.tif'))
    manual_segs = io.imread(path.join(dirname,'manual_tracking/manual_tracking_final.tiff'))
    for t in tqdm(range(manual_segs.shape[0])):
        manual_segs[t,...] = segmentation.expand_labels(manual_segs[t,...],distance=dist2expand) 
    
    io.imsave(path.join(dirname,f'manual_tracking/manual_tracking_final_exp{dist2expand}.tiff'))
    
    G = io.imread(path.join(dirname,'master_stack/G.tif'))
    
    G_th = np.zeros_like(G,dtype=bool)
    
    kernel_size = (G.shape[1] // 3,
                   G.shape[2] // 8,
                   G.shape[3] // 8)
    kernel_size = np.array(kernel_size)
    
    if not path.exists(path.join(dirname,'master_stack/G_clahe.tif')):
        G_clahe = np.zeros_like(G,dtype=float)
        for t, im_time in tqdm(enumerate(G)):
            G_clahe[t,...] = exposure.equalize_adapthist(im_time, kernel_size=kernel_size, clip_limit=0.01, nbins=256)
            for z, im in enumerate(G_clahe[t,...]):
                G_th[t,z,...] = im > filters.threshold_otsu(im)
        io.imsave(path.join(dirname,'master_stack/G_clahe.tif'),util.img_as_uint(G_clahe))
    else:    
        G_clahe = io.imread(path.join(dirname,'master_stack/G_clahe.tif'))
        for t, im_time in tqdm(enumerate(G)):
            for z, im in enumerate(G_clahe[t,...]):
                G_th[t,z,...] = im > filters.threshold_otsu(im)
    io.imsave(path.join(dirname,'master_stack/G_clahe_th.tif'),G_th)
    
    


  