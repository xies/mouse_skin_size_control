#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:55:26 2022

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
from mathUtils import normxcorr2
import matplotlib.pylab as plt

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'

#%% Reading the first ome-tiff file using imread reads entire stack

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
# B_tifs = sorted(glob(path.join(dirname,'*Day*/B_reg_reg.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*Day*/G_reg.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*Day*/R_shg_reg_reg.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*Day*/R_reg_reg.tif')),key=sort_by_day)

assert(len(G_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

manual_Ztarget = {}

#%% Correlate each R_shg timepoint with first time point
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024

OVERWRITE = True

XY_reg = True
manual_Ztarget = {}
APPLY = True

ref_T = 0


# Find the slice with maximum mean value in R_shg channel
R_shg_ref = io.imread( R_shg_tifs[ref_T] )
Z_ref = R_shg_ref.shape[ref_T]
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = R_shg_ref[Imax_ref,...]


# variables to save:
z_pos_in_original = np.zeros(len(G_tifs))
z_pos_in_original[ref_T] = Imax_ref
XY_matrices = np.zeros((len(G_tifs),3,3))

# for t in tqdm( np.arange(0,len(G_tifs)) ): # 0-indexed
t = 14
# if t == ref_T:
#     continue

output_dir = path.split(path.dirname(R_tifs[t]))[0]
# if APPLY and not OVERWRITE and path.exists(path.join(path.dirname(G_tifs[t]),'G_align.tif')):
#     print(f'Skipping t = {t}')
#     continue

print(f'Working on {R_shg_tifs[t]}')
#Load the target
R_shg_target = io.imread(R_shg_tifs[t]).astype(float)

# Find simlar in the next time point
# If specified, use the manually determined ref_z
if t in manual_Ztarget.keys():
    Imax_target = manual_Ztarget[t]
    print(f'Target z-slice manually set at {Imax_target}')
else:
    # 1. Use xcorr2 to find the z-slice on the target that has max CC with the reference
    CC = np.zeros(R_shg_target.shape[0])
    for z,im in enumerate(R_shg_target):
        CC[z] = normxcorr2(ref_img, im).max()
    Imax_target = CC.argmax()
    print(f'Target z-slice automatically determined to be {Imax_target}')
z_pos_in_original[t] = Imax_target

# Perform transformations
# B = io.imread(B_tifs[t])
G = util.img_as_float(io.imread(G_tifs[t]))
R = util.img_as_float(io.imread(R_tifs[t]))

# B_transformed = B.copy();
R_transformed = R.copy(); G_transformed = G.copy(); R_shg_transformed = R_shg_target.copy();

if XY_reg:
    moving_img = R_shg_target[Imax_target,...]
    print('\n Starting stackreg')
    # 2. Use StackReg to 'align' the two z slices
    sr = StackReg(StackReg.RIGID_BODY)
    T = sr.register(ref_img,moving_img) #Obtain the transformation matrices
    XY_matrices[t,...] = T
    
    if APPLY:
        print('Applying transformation matrices')
        # Apply transformation matrix to each stacks
        
        for i, G_slice in enumerate(G):
            # B_transformed[i,...] = sr.transform(B_slice.astype(float),tmat=T)
            G_transformed[i,...] = sr.transform(G_slice,tmat=T)
        for i, R_slice in enumerate(R):
            R_transformed[i,...] = sr.transform(R_slice,tmat=T)
            R_shg_transformed[i,...] = sr.transform(R_shg_target[i,...],tmat=T)
    
if APPLY:    
    # Z-pad the time point in reference to t - 1
    Z_target = R_shg_target.shape[0]

    print('Padding')
    top_padding = Imax_ref - Imax_target
    if top_padding > 0: # the needs padding
        R_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_transformed), axis= 0)
        G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
        # B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
        R_shg_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_shg_transformed), axis= 0)
        
    elif top_padding < 0: # then needs trimming
        R_padded = R_transformed[-top_padding:,...]
        G_padded = G_transformed[-top_padding:,...]
        # B_padded = G_transformed[-top_padding:,...]
        R_shg_padded = R_shg_transformed[-top_padding:,...]
        
    elif top_padding == 0:
        R_padded = R_transformed
        G_padded = G_transformed
        # B_padded = B_transformed
        R_shg_padded = R_shg_transformed
        
    delta_ref = Z_ref - Imax_ref
    delta_target = Z_target - Imax_target
    bottom_padding = delta_ref - delta_target
    if bottom_padding > 0: # the needs padding
        R_padded = np.concatenate( (R_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
        G_padded = np.concatenate( (G_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
        # B_padded = np.concatenate( (B_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
        R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
        
    elif bottom_padding < 0: # then needs trimming
        R_padded = R_padded[0:bottom_padding,...]
        G_padded = G_padded[0:bottom_padding,...]
        # B_padded = B_padded[0:bottom_padding,...]
        R_shg_padded = R_shg_padded[0:bottom_padding,...]

    print('Saving')
    output_dir = path.dirname(R_tifs[t])
    # io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.int16),check_contrast=False)
    io.imsave(path.join(output_dir,'G_align.tif'),util.img_as_uint(G_padded/G_padded.max()),check_contrast=False)
    
    output_dir = path.dirname(R_tifs[t])
    io.imsave(path.join(output_dir,'R_align.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
    io.imsave(path.join(output_dir,'R_shg_align.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)


# import pickle as pkl
# with open(path.join(dirname,'alignment_information.pkl'),'wb') as f:
#     print('Saving alignment matrices...')
#     pkl.dump([z_pos_in_original,XY_matrices,Imax_ref],f)

# print('DONE')

#%%
#% Manually input any alignment matrix and save

# t = 5

# XY_matrices[ref_T] = np.eye(3)

# T = np.array([[np.cos(-2.5),-np.sin(-2.5),40],
#               [np.sin(-2.5),np.cos(-2.5),35],
#               [0,0,1]])
# XY_matrices[t] = T

# z_pos_in_original[t] = 16


import pickle as pkl
with open(path.join(dirname,'alignment_information.pkl'),'wb') as f:
    print('Saving alignment matrices...')
    pkl.dump([z_pos_in_original,XY_matrices,Imax_ref],f)

print('DONE')

#%% Sort filenames by time (not alphanumeric) and then assemble each time point
        
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = len(G_tifs)

filelist = pd.DataFrame()
# filelist['B'] = sorted(glob(path.join(dirname,'*Day*/ZSeries*/B_align.tif')), key = sort_by_day)
filelist['G'] = sorted(glob(path.join(dirname,'*Day*/G_align.tif')), key = sort_by_day)
filelist['R'] = sorted(glob(path.join(dirname,'*Day*/R_align.tif')), key = sort_by_day)
filelist['R_shg'] = sorted(glob(path.join(dirname,'*Day*/R_shg_align.tif')), key = sort_by_day)
filelist.index = np.arange(1,T)

# t= 0 has no '_align'
s = pd.Series({
                  'G': glob(path.join(dirname,'*Day 0.5/G_reg_reg.tif'))[0],
                  'R': glob(path.join(dirname,'*Day 0.5/R_reg_reg.tif'))[0],
                   'R_shg': glob(path.join(dirname,'*Day 0.5/R_shg_reg_reg.tif'))[0]}
              , name=0)

filelist = filelist.append(s)
filelist = filelist.sort_index()

# Save individual day*.tif

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
    # B = io.imread(filelist.loc[t,'B'])
    R_ = fix_image_range(R,MAX)
    G_ = fix_image_range(G,MAX)
# B_ = fix_image_range(B,MAX)

# Do some image range clean up

    stack = np.stack((R_,G_))
    io.imsave(path.join(dirname,f'im_seq/t{t}.tif'),stack.astype(np.uint16),check_contrast=False)

#%% Save master stack
# Load file and concatenate them appropriately
# FIJI default: CZT XY, but this is easier for indexing
stack = np.zeros((T,Z_ref,3,XX,XX))

for t in range(T):
    R = io.imread(filelist.loc[t,'R'])
    G = io.imread(filelist.loc[t,'G'])
    # B = io.imread(filelist.loc[t,'B'])
    
    stack[t,:,0,:,:] = R
    stack[t,:,1,:,:] = G
    # stack[t,:,2,:,:] = B
    
# io.imsave(path.join(dirname,'master_stack.tif'),stack.astype(np.int16))



