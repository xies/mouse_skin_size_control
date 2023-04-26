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
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'

#%%

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

#%%

filelist = pd.DataFrame()
# filelist['B'] = sorted(glob(path.join(dirname,'*Day*/B_align.tif')), key = sort_by_day)
filelist['G'] = sorted(glob(path.join(dirname,'*Day*/G_align.tif')), key = sort_by_day)
filelist['R'] = sorted(glob(path.join(dirname,'*Day*/R_align.tif')), key = sort_by_day)
filelist['R_shg'] = sorted(glob(path.join(dirname,'*Day*/R_shg_align.tif')), key = sort_by_day)
filelist.index = np.arange(1,len(filelist)+1)

# t= 0 has no '_align'
s = pd.DataFrame({
                  # 'B': glob(path.join(dirname,'*Day 0/B_reg.tif'))[0],
                  'G': glob(path.join(dirname,'*Day 0/G_reg.tif'))[0],
                  'R': glob(path.join(dirname,'*Day 0/R_reg_reg.tif'))[0],
                   'R_shg': glob(path.join(dirname,'*Day 0.5/R_shg_reg_reg.tif'))[0]},index=[0])

filelist = pd.concat([filelist,s])
filelist = filelist.sort_index()

#%% Stack channels & save into single tif

G = np.zeros((len(filelist),80,1024,1024))
for t in tqdm(range(16)):
    
    G_ = io.imread(filelist.loc[t,'G'])
    G[t,...] = G_
    # R = io.imread(R_tifs[t])
    # R_shg = io.imread(R_shg_tifs[t])
    
    # stack = np.stack((G,R,R_shg),axis=0)
    
    d = path.split( filelist.loc[t,'G'] )[0]
    
io.imsave(path.join(d,f'im_seq/t{t}.tif'), G)
    
    