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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-18-2023 R26CreER Rb-fl ablation test/F1 black/R1'

#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
# filelist = parse_unaligned_channels(dirname)
       
#%% Correlate each R_shg timepoint with subsequent timepoint (Nope, using first time point instead)
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024
OVERWRITE = True

assert(len(B_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

ref_T = 2
target_T = 9

R_shg_ref = io.imread( R_shg_tifs[3] )

###

# Find the slice with maximum mean value in R_shg channel
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = R_shg_ref[Imax_ref,...]
Z_ref = R_shg_ref.shape[0]

CC = np.zeros(len(R_tifs))
Iz_target_slice = np.zeros(len(R_tifs))

output_dir = path.split(path.dirname(R_tifs[target_T]))[0]

#Load the target
# R_shg_target = io.imread(R_shg_tifs[target_T])

# Find simlar in the next time point
Imax_target = R_shg_target.std(axis=2).std(axis=1).argmax()
target_img = R_shg_target[Imax_target,...]

print('\n Starting stackreg')
# Use StackReg to 'align' the two z slices
sr = StackReg(StackReg.RIGID_BODY)
T = sr.register(ref_img,target_img) #Obtain the transformation matrices

B = io.imread(B_tifs[target_T])
G = io.imread(G_tifs[target_T])
# R = io.imread(R_tifs[target_T])

T = transform.SimilarityTransform(matrix=T)

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
Z_target = R_shg_target.shape[0]

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
output_dir = path.dirname(B_tifs[target_T])
io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.int16))
io.imsave(path.join(output_dir,'G_align.tif'),G_padded.astype(np.int16))

# output_dir = path.dirname(R_tifs[target_T])
# io.imsave(path.join(output_dir,'R_align.tif'),R_padded.astype(np.int16))
# io.imsave(path.join(output_dir,'R_shg_align.tif'),R_shg_padded.astype(np.int16))
    
#%% Sort filenames by time (not alphanumeric) and then assemble 'master stack'
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = len(B_tifs)-1
# Use a function to regex the Day number and use that to sort
def sort_by_number(filename):
    day = match('Day (\d+)',path.split(path.split(path.split(filename)[0])[0])[1])
    day = day.groups()[0]
    return int(day)

filelist = pd.DataFrame()
filelist['B'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/B_align.tif')), key = sort_by_number)
filelist['G'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/G_align.tif')), key = sort_by_number)
filelist['R'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/R_align.tif')), key = sort_by_number)
filelist.index = np.arange(1,T+1)

# t= 0 has no '_align'
s = pd.Series({'B': glob(path.join(dirname,'Day 3.5/ZSeries*/B_reg_reg.tif'))[0],
                 'G': glob(path.join(dirname,'Day 3.5/ZSeries*/G_reg_reg.tif'))[0],
                 'R': glob(path.join(dirname,'Day 3.5/ZSeries*/R_reg_reg.tif'))[0]}, name=0)

filelist = filelist.append(s)
filelist = filelist.sort_index()

# Load file and concatenate them appropriately
# FIJI default: CZT XY, but this is easier for indexing
stack = np.zeros((T,Z_ref,3,XX,XX))

for t in range(T):
    R = io.imread(filelist.loc[t,'R'])
    G = io.imread(filelist.loc[t,'G'])
    B = io.imread(filelist.loc[t,'B'])
    
    stack[t,:,0,:,:] = R
    stack[t,:,1,:,:] = G
    stack[t,:,2,:,:] = B
    
io.imsave(path.join(dirname,'master_stack.tif'),stack.astype(np.int16))



