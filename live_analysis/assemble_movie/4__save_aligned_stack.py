#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 16:53:19 2022

@author: xies
"""


import numpy as np
import pandas as pd
from skimage import io, transform, filters, util
from os import path
from re import match
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm
import matplotlib.pylab as plt

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R1'

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
                   'R_shg': glob(path.join(dirname,'*Day 0/R_shg_reg_reg.tif'))[0]},index=[0])

filelist = pd.concat([filelist,s])
filelist = filelist.sort_index()

#%% Stack channels & save into single tif

G = []
# R = []

for t in tqdm(range(16)):
    
    G_ = io.imread(filelist.loc[t,'G'])
    G.append(G_)
    # R_ = io.imread(filelist.loc[t,'R'])
    # R.append(R_)
    
    # stack = np.stack((G,R,R_shg),axis=0)
    

io.imsave(path.join(dirname,f'master_stack/G.tif'), np.stack(G).astype(np.uint16))
# io.imsave(path.join(dirname,f'master_stack/R.tif'), np.stack(R).astype(np.uint16))
   

#%% Sort filenames by time (not alphanumeric) and then assemble each time point
        
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = len(G_tifs)

filelist = pd.DataFrame()
filelist['B'] = sorted(glob(path.join(dirname,'*Day*/B_align.tif')), key = sort_by_day)
filelist['G'] = sorted(glob(path.join(dirname,'*Day*/G_align.tif')), key = sort_by_day)
filelist['R'] = sorted(glob(path.join(dirname,'*Day*/R_align.tif')), key = sort_by_day)
filelist['R_shg'] = sorted(glob(path.join(dirname,'*Day*/R_shg_align.tif')), key = sort_by_day)
filelist.index = np.arange(1,T)

# t= 0 has no '_align'
s = pd.DataFrame({'B': glob(path.join(dirname,'*Day 0/B_reg.tif'))[0],
                  'G': glob(path.join(dirname,'*Day 0/G_reg.tif'))[0],
                  'R': glob(path.join(dirname,'*Day 0/R_reg_reg.tif'))[0],
                   'R_shg': glob(path.join(dirname,'*Day 0.5/R_shg_reg_reg.tif'))[0]},index=[0])

filelist = pd.concat([filelist,s])
filelist = filelist.sort_index()

#%% Save individual day*.tif
MAX = 2**16-1
def fix_image_range(im, max_range):
    
    im = im.copy().astype(float)
    im[im == 0] = np.nan
    im = im - np.nanmin(im)
    im = im / np.nanmax(im) * max_range
    im[np.isnan(im)] = 0
    return im.astype(np.uint16)

for t in tqdm(range(T)):

# stack = np.zeros((Z_ref,3,XX,XX))
    
    R = io.imread(filelist.loc[t,'R'])
    G = io.imread(filelist.loc[t,'G'])
    B = io.imread(filelist.loc[t,'B'])
    
    # Do some image range clean up
    R_ = fix_image_range(R,MAX)
    G_ = fix_image_range(G,MAX)
    B_ = fix_image_range(B,MAX)

    stack = np.stack((R_,G_,B_))
    io.imsave(path.join(dirname,f'im_seq/t{t}.tif'),stack.astype(np.uint16),check_contrast=False)

#%% Save master stack
# Load file and concatenate them appropriately
# FIJI default: CZT XY, but this is easier for indexing
R = np.zeros((T,Z_ref,XX,XX))
G = np.zeros((T,Z_ref,XX,XX))

for t in tqdm(range(T)):
    # R_ = io.imread(filelist.loc[t,'R'])
    G_ = io.imread(filelist.loc[t,'G'])
    # B_ = io.imread(filelist.loc[t,'B'])
    
    # R[t,...] = R_
    G[t,...] = G_
    # B[t,...] = B_
    
# io.imsave(path.join(dirname,'R.tif'),util.img_as_uint(R/R.max()))
io.imsave(path.join(dirname,'G.tif'),util.img_as_uint(G/G.max()))




