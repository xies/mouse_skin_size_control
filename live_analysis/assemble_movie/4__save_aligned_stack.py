#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:53:19 2022

@author: xies
"""


import numpy as np
import pandas as pd
from skimage import io, transform, filters
from os import path
from re import match
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm
import matplotlib.pylab as plt

dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'

#%%

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
# B_tifs = sorted(glob(path.join(dirname,'*Day*/ZSeries*/B_align.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*Day*/G_align.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*Day*/R_shg_align.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*Day*/R_align.tif')),key=sort_by_day)

#%% Stack channels & save into single tif

for t in tqdm(range(17)):
    
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    R_shg = io.imread(R_shg_tifs[t])
    
    stack = np.stack((G,R,R_shg),axis=0)
    
    d = path.split( path.dirname(G_tifs[t]))[0]
    
    io.imsave(path.join(d,f'im_seq/t{t}.tif'), stack)
    
    