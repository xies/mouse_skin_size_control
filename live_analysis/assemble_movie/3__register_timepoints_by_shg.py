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
import pickle as pkl

from twophotonUtils import parse_unaligned_channels

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2/'

#%% Reading the first ome-tiff file using imread reads entire stack

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

filelist = parse_unaligned_channels(dirname,folder_str='*.*/')

# Grab all registered B/R tifs
# B_tifs = sorted(glob(path.join(dirname,'*Day*/B_reg.tif')),key=sort_by_day)
# G_tifs = sorted(glob(path.join(dirname,'*Day*/G_reg.tif')),key=sort_by_day)
# R_shg_tifs = sorted(glob(path.join(dirname,'*Day*/R_shg_reg_reg.tif')),key=sort_by_day)
# R_tifs = sorted(glob(path.join(dirname,'*Day*/R_reg_reg.tif')),key=sort_by_day)

# assert(len(G_tifs) == len(R_tifs))
# assert(len(G_tifs) == len(R_shg_tifs))

manual_Ztarget = {}

#%% 

XX = 1024
TT = len(filelist)

OVERWRITE = True

XY_reg = True
manual_Ztarget = {1:60,2:49,3:51,4:63,5:56}
APPLY_XY = True
APPLY_PAD = True

ref_T = 0

z_pos_in_original = {}
XY_matrices = {}
if path.exists(path.join(dirname,'alignment_information.pkl')):
    with open(path.join(dirname,'alignment_information.pkl'),'rb') as f:
        [z_pos_in_original,XY_matrices,Imax_ref] = pkl.load(f)

# Find the slice with maximum mean value in R_shg channel
R_shg_ref = io.imread( filelist.loc[ref_T,'R_shg'] )
Z_ref = R_shg_ref.shape[ref_T]
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
# Imax_ref = 31
ref_img = R_shg_ref[Imax_ref,...]
print(f'Reference z-slice: {Imax_ref}')

# variables to save:
z_pos_in_original[ref_T] = Imax_ref

#%% Z-target and XY transform
# Correlate each R_shg timepoint with first time point
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

for t in tqdm( [5] ): # 0-indexed
    if t == ref_T:
        continue
    
    output_dir = path.split(path.dirname(filelist.loc[t,'R']))[0]
    if not OVERWRITE and path.exists(path.join(path.dirname(filelist.loc[t,'B']),'B_align.tif')):
        print(f'\n Skipping t = {t}')
        continue
    
    print(f'\n Working on t = {t}')
    #Load the target
    R_shg_target = io.imread(filelist.loc[t,'R_shg']).astype(float)
    
    # Find simlar in the next time point
    # If specified, use the manually determined ref_z
    if t in z_pos_in_original.keys() and not OVERWRITE:
        Imax_target = z_pos_in_original[t]
        print(f'Target z-slice is pre-defined at {Imax_target}')
    else:
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
    B = util.img_as_float(io.imread(filelist.loc[t,'B']))
    G = util.img_as_float(io.imread(filelist.loc[t,'G']))
    R = util.img_as_float(io.imread(filelist.loc[t,'R']))
    
    B_transformed = B.copy();
    R_transformed = R.copy(); G_transformed = G.copy(); R_shg_transformed = R_shg_target.copy();
    
    if XY_reg:
        if t in XY_matrices.keys() and not OVERWRITE:
            print('\n XY registration is pre-defined')
            T = XY_matrices[t]
        else:
            moving_img = R_shg_target[Imax_target,...]
            print('\n Starting stackreg')
            # 2. Use StackReg to 'align' the two z slices
            sr = StackReg(StackReg.RIGID_BODY)
            T = sr.register(ref_img,moving_img) #Obtain the transformation matrices
            XY_matrices[t] = T
        
        if APPLY_XY:
            print('Applying transformation matrices')
            # Apply transformation matrix to each stacks
            
            T = transform.SimilarityTransform(T)
            # T = T + transform.SimilarityTransform(translation=[-10,-40],rotation=np.deg2rad(-1))
            
            for i, G_slice in enumerate(G):
                B_transformed[i,...] = transform.warp(B[i,...].astype(float),T)
                G_transformed[i,...] = transform.warp(G_slice,T)
            for i, R_slice in enumerate(R):
                R_transformed[i,...] = transform.warp(R_slice,T)
                R_shg_transformed[i,...] = transform.warp(R_shg_target[i,...],T)
        
    if APPLY_PAD:
        # Z-pad the time point in reference to t - 1
        Z_target = R_shg_target.shape[0]
    
        print('Padding')
        top_padding = Imax_ref - Imax_target
        if top_padding > 0: # the needs padding
            R_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_transformed), axis= 0)
            G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
            B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
            R_shg_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_shg_transformed), axis= 0)
            
        elif top_padding < 0: # then needs trimming
            R_padded = R_transformed[-top_padding:,...]
            G_padded = G_transformed[-top_padding:,...]
            B_padded = B_transformed[-top_padding:,...]
            R_shg_padded = R_shg_transformed[-top_padding:,...]
            
        elif top_padding == 0:
            R_padded = R_transformed
            G_padded = G_transformed
            B_padded = B_transformed
            R_shg_padded = R_shg_transformed
            
        delta_ref = Z_ref - Imax_ref
        delta_target = Z_target - Imax_target
        bottom_padding = delta_ref - delta_target
        if bottom_padding > 0: # the needs padding
            R_padded = np.concatenate( (R_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            G_padded = np.concatenate( (G_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            B_padded = np.concatenate( (B_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
            
        elif bottom_padding < 0: # then needs trimming
            R_padded = R_padded[0:bottom_padding,...]
            G_padded = G_padded[0:bottom_padding,...]
            B_padded = B_padded[0:bottom_padding,...]
            R_shg_padded = R_shg_padded[0:bottom_padding,...]
    
        print('Saving')
        output_dir = path.dirname(filelist.loc[t,'G'])
        io.imsave(path.join(output_dir,'B_align.tif'),util.img_as_uint(B_padded/B_padded.max()),check_contrast=False)
        io.imsave(path.join(output_dir,'G_align.tif'),util.img_as_uint(G_padded/G_padded.max()),check_contrast=False)
        
        output_dir = path.dirname(filelist.loc[t,'R'])
        io.imsave(path.join(output_dir,'R_align.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
        io.imsave(path.join(output_dir,'R_shg_align.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)
    

with open(path.join(dirname,'alignment_information.pkl'),'wb') as f:
    print('Saving alignment matrices...')
    pkl.dump([z_pos_in_original,XY_matrices,Imax_ref],f)

print('DONE')

#%%
#% Manually input any alignment matrix and save

# t = 5

# XY_matrices[ref_T] = np.eye(3)

# T = np.array([[np.cos(-2.5),-np.sin(-2.5),40],
#               [np.sin(-2.5),np.cos(-2.5),35],
#               [0,0,1]])
# XY_matrices[t] = T

# z_pos_in_original[t] = 16


# import pickle as pkl
# with open(path.join(dirname,'alignment_information.pkl'),'wb') as f:
#     print('Saving alignment matrices...')
#     pkl.dump([z_pos_in_original,XY_matrices,Imax_ref],f)

# print('DONE')

