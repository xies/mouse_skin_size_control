#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:46:46 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, filters,exposure, util
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

from basicUtils import *

dirnames = {}
# dirnames['WT1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
# dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
# dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M1 WT/R1'

# dirnames['RBKO1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M6 RBKO/R1'

#%% name

Zref = {}
Zref['RBKO1'] = 31
Zref['WT2'] = 34

for name,dirname in tqdm(dirnames.items()):
    
    G = io.imread(path.join(dirname,'master_stack/G.tif'))
    
    G_eq = np.zeros(G.shape,dtype=float)
    for t,im in tqdm(enumerate(G)):
        im = im[Zref[name],...]
        im_nonzero = im[im>0]
        G_eq[t,...] = G[t,...] / im_nonzero.mean()

    G_eq[G_eq > 5] = 5
    
    io.imsave(path.join(dirname,'master_stack/G_eq.tif'),util.img_as_uint(G_eq/5))
    

#%%

for name,dirname in tqdm(dirnames.items()):
    
    G = io.imread(path.join(dirname,'master_stack/G_eq.tif'))
    
    th = filters.threshold_otsu(G)
    # for t,im in tqdm(enumerate(G)):
    #     th = filters.threshold_otsu(im)
    G_th = G > th
        
    io.imsave(path.join(dirname,'master_stack/G_th.tif'), util.img_as_ubyte(G_th))
    
    