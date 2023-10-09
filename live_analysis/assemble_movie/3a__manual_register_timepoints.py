#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:29:22 2022

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

from twophotonUtils import parse_unaligned_channels
from mathUtils import normxcorr2

# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-18-2023 R26CreER Rb-fl ablation test/F1 black/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/10-04-2023 R26CreER Rb-fl no tam ablation M5/M5 white DOB 4-25-23/R1'

filelist = parse_unaligned_channels(dirname,folder_str='*.*/')

#%% Specify source and target

XX = 1024
OVERWRITE = True

ref_T = 1
target_T = 0

ref_chan = 'B'
target_chan = 'B'


#Load the ref
ref_stack = io.imread( filelist.loc[ref_T,ref_chan] )
#Load the target
target_stack = io.imread(filelist.loc[target_T,target_chan])

#%%

# Find the slice with maximum mean value in R_shg channel
Imax_ref = ref_stack.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = ref_stack[Imax_ref,...]
Z_ref = ref_stack.shape[0]

# Use xcorr2 to find the z-slice on the target that has max CC with the reference
CC = np.zeros(target_stack.shape[0])
for z,im in enumerate(target_stack):
    CC[z] = normxcorr2(ref_img, im).max()
Imax_target = CC.argmax()
target_img = target_stack[Imax_target]
print(f'Target z-slice automatically determined to be {Imax_target}')


print('\n Starting stackreg')
# Use StackReg to 'align' the two z slices
sr = StackReg(StackReg.RIGID_BODY)
T = sr.register(ref_img,target_img) #Obtain the transformation matrices

B = io.imread(filelist.loc[target_T,'B'])
G = io.imread(filelist.loc[target_T,'G'])
# R = io.imread(R_tifs[target_T])


print('Applying transformation matrices')
# Apply transformation matrix to each stacks
B_transformed = np.zeros_like(B)
G_transformed = np.zeros_like(G)
# R_transformed = np.zeros_like(R)
# R_shg_transformed = np.zeros_like(R)
for i, B_slice in enumerate(B):
    B_transformed[i,...] = transform.warp(B_slice.astype(float),T)
    G_transformed[i,...] = transform.warp(G[i,...].astype(float),T)
    # R_transformed[i,...] = transform.warp(R[i,...].astype(float),T)
    # R_shg_transformed[i,...] = transform.warp(R_shg_target[i,...].astype(float),T)
    
# Z-pad the time point in reference to t - 1
Z_target = target_stack.shape[0]

print('Padding')
top_padding = Imax_ref - Imax_target
if top_padding > 0: # the needs padding
    # R_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_transformed), axis= 0)
    G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
    B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
    # R_shg_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_shg_transformed), axis= 0)
    
elif top_padding < 0: # then needs trimming 
    # R_padded = R_transformed[-top_padding:,...]
    G_padded = G_transformed[-top_padding:,...]
    B_padded = B_transformed[-top_padding:,...]
    # R_shg_padded = R_shg_transformed[-top_padding:,...]
    
elif top_padding == 0:
    # R_padded = R_transformed
    G_padded = G_transformed
    B_padded = B_transformed
    # R_shg_padded = R_shg_transformed
    
delta_ref = Z_ref - Imax_ref
delta_target = Z_target - Imax_target
bottom_padding = delta_ref - delta_target
if bottom_padding > 0: # the needs padding
    # R_padded = np.concatenate( (R_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    G_padded = np.concatenate( (G_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    B_padded = np.concatenate( (B_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    # R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    
elif bottom_padding < 0: # then needs trimming
    # R_padded = R_padded[0:bottom_padding,...]
    G_padded = G_padded[0:bottom_padding,...]
    B_padded = B_padded[0:bottom_padding,...]
    # R_shg_padded = R_shg_padded[0:bottom_padding,...]
    
print('Saving')
output_dir = path.dirname(filelist.loc[target_T,'B'])
io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.uint16))
io.imsave(path.join(output_dir,'G_align.tif'),G_padded.astype(np.uint16))

# output_dir = path.dirname(R_tifs[target_T])
# io.imsave(path.join(output_dir,'R_align.tif'),R_padded.astype(np.int16))
# io.imsave(path.join(output_dir,'R_shg_align.tif'),R_shg_padded.astype(np.int16))
    



