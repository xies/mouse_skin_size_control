#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:55:26 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, filters, util, transform
from os import path
from re import match
from glob import glob
from pystackreg import StackReg
from re import findall
from tqdm import tqdm

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/03-24-2022 power series 24h/M8 WT/R5 940nm_pw150 1020nm_pw225'

#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
B_tifs = glob(path.join(dirname,'Day*/ZSeries*/B_reg_reg.tif'))
G_tifs = glob(path.join(dirname,'Day*/ZSeries*/G_reg_reg.tif'))
R_shg_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_shg_reg_reg.tif'))
R_tifs = glob(path.join(dirname,'Day*/ZSeries*/R_reg_reg.tif'))

# @todo: need to sort

#%% Correlate each R_shg timepoint with subsequent timepoint.
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024

assert(len(B_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

R_shg_ref = io.imread( R_shg_tifs[0] )

# Find the slice with maximum mean value in R_shg channel
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
ref_img = R_shg_ref[Imax_ref,...] # Don't gaussian filter for stackreg
Z_ref = R_shg_ref.shape[0]

for t in tqdm(np.arange(1,len(R_tifs) )): # 1-indexed + progress
    
    # Reference is always first image
    R_shg_target = io.imread(R_shg_tifs[t])
    
    # Find simlar in the next time point
    Imax_target = R_shg_target.std(axis=2).std(axis=1).argmax()
    target_img = R_shg_target[Imax_target,...]
    
    # Use StackReg to 'align' the two z slices
    sr = StackReg(StackReg.RIGID_BODY) # There should only be slight sliding motion within a single stack
    T = sr.register(ref_img,target_img) #Obtain the transformation matrices
    
    B = io.imread(B_tifs[t])
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    
    T = transform.SimilarityTransform(matrix=T)
    
    # Apply transformation matrix to each stacks
    B_transformed = np.zeros_like(B)
    G_transformed = np.zeros_like(G)
    R_transformed = np.zeros_like(R)
    R_shg_transformed = np.zeros_like(R)
    for i, B_slice in enumerate(B):
        B_transformed[i,...] = transform.warp(B_slice.astype(float),T)
        G_transformed[i,...] = transform.warp(G[i,...].astype(float),T)
        R_transformed[i,...] = transform.warp(R[i,...].astype(float),T)
        R_shg_transformed[i,...] = transform.warp(R_shg_target[i,...].astype(float),T)
    
    # @todo: padding and stuff
    # Z-pad the time point in reference to t = 0
    Z_target = R_shg_target.shape[0]
    
    top_padding = Imax_ref - Imax_target
    if top_padding > 0: # the needs padding
        R_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_transformed), axis= 0)
        G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
        B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
        R_shg_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),R_shg_transformed), axis= 0)
        
    elif top_padding < 0: # then needs trimming
        R_padded = R[-top_padding:,...]
        G_padded = G[-top_padding:,...]
        B_padded = B[-top_padding:,...]
        R_padded = R_shg_target[-top_padding:,...]
        
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
        
    
    output_dir = path.dirname(B_tifs[t])
    io.imsave(path.join(output_dir,'B_align.tif'),B_padded.astype(np.int16))
    io.imsave(path.join(output_dir,'G_align.tif'),G_padded.astype(np.int16))
    
    output_dir = path.dirname(R_tifs[t])
    io.imsave(path.join(output_dir,'R_align.tif'),R_padded.astype(np.int16))
    io.imsave(path.join(output_dir,'R_shg_align.tif'),R_shg_padded.astype(np.int16))
    
#%% Sort filenames by time (not alphanumeric) and then assemble 'master stack'
# But exclude R_shg since 4-channel tifs are annoying to handle for FIJI loading.

T = 9
# Use a function to regex the Day number and use that to sort
def sort_by_number(filename):
    day = match('Day (\d+)',path.split(path.split(path.split(filename)[0])[0])[1])
    day = day.groups()[0]
    return int(day)

filelist = pd.DataFrame()
filelist.index = np.arange(1,T)
filelist['B'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/B_align.tif')), key = sort_by_number)
filelist['G'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/G_align.tif')), key = sort_by_number)
filelist['R'] = sorted(glob(path.join(dirname,'Day*/ZSeries*/R_align.tif')), key = sort_by_number)

# t= 0 has no '_align'
s = pd.Series({'B': glob(path.join(dirname,'Day 0/ZSeries*/B_reg_reg.tif'))[0],
                 'G': glob(path.join(dirname,'Day 0/ZSeries*/G_reg_reg.tif'))[0],
                 'R': glob(path.join(dirname,'Day 0/ZSeries*/R_reg_reg.tif'))[0]}, name=0)

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



