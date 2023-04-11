#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 21:41:31 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, util
from os import path
from glob import glob
from re import match
from tqdm import tqdm
from pystackreg import StackReg
from mathUtils import normxcorr2

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/M1 WT/R1'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R2'

#%% Reading the first ome-tiff file using imread reads entire stack

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
B_tifs = sorted(glob(path.join(dirname,'*. Day*/B_reg.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*. Day*/G_reg.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*. Day*/R_shg_reg.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*. Day*/R_reg.tif')),key=sort_by_day)

#%%

XX = 1024

OVERWRITE = False

# assert(len(B_tifs) == len(R_tifs))

manual_targetZ = {7:27}
for t in range(len(R_tifs)):
# t = 6
    
    output_dir = path.split(path.dirname(R_tifs[t]))[0]
    if path.exists(path.join(path.dirname(R_tifs[t]),'R_reg_reg.tif'))  and not OVERWRITE:
    # and path.exists(path.join(path.dirname(B_tifs[t]),'B_reg_reg.tif'))  and not OVERWRITE:
        print(f'Skipping t = {t} because its R_reg_reg.tif already exists')
        continue
    
    print(f'\n--- Started t = {t} ---')
    B = io.imread(B_tifs[t])
    R_shg = io.imread(R_shg_tifs[t])
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    
    print('Done reading images')
    
    # Find the slice with maximum mean value in R_shg channel
    Imax = R_shg.mean(axis=2).mean(axis=1).argmax()
    R_ref = R_shg[Imax,...]
    R_ref = filters.gaussian(R_ref,sigma=0.5)
    
    if t in manual_targetZ.keys():
        Iz = manual_targetZ[t]
        print(f'Manually setting target Z-slice at : {Iz}')
    else:
        # Iteratively find maximum x-corr (2D) for each B channel slice
        
        CC = np.zeros((B.shape[0],XX * 2 - 1,XX * 2 -1))
        
        print('Cross correlation started')
        for i,B_slice in enumerate(B):
            B_slice = filters.gaussian(B_slice,sigma=0.5)
            CC[i,...] = normxcorr2(R_ref,B_slice,mode='full')
        [Iz,y_shift,x_shift] = np.unravel_index(CC.argmax(),CC.shape) # Iz refers to B channel
        print(f'Cross correlation done and target Z-slice set at: {Iz}')
        
    target = filters.gaussian(B[Iz,...],sigma=0.5)
    
    #NB: Here, move the R channel wrt the B channel
    print('StackReg + transform')
    sr = StackReg(StackReg.RIGID_BODY)
    T = sr.register(target/target.max(),R_ref) #Obtain the transformation matrices
    
    R_transformed = np.zeros_like(R).astype(float)
    R_shg_transformed = np.zeros_like(R).astype(float)
    for i, R_slice in enumerate(R):
        R_transformed[i,...] = sr.transform(R_slice,tmat=T)
        R_shg_transformed[i,...] = sr.transform(R_shg[i,...],tmat=T)
    
    
    output_dir = path.dirname(B_tifs[t])
    # io.imsave(path.join(output_dir,'B_reg_reg.tif'),B_transformed.astype(np.int16),check_contrast=False)
    # io.imsave(path.join(output_dir,'G_reg_reg.tif'),G_transformed.astype(np.int16),check_contrast=False)
    
    print('Padding')
    # Z-pad the red + red_shg channel using Imax and Iz
    bottom_padding = Iz - Imax
    if bottom_padding > 0: # the needs padding
        R_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_transformed), axis= 0)
        R_shg_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_shg_transformed), axis= 0)
    elif bottom_padding < 0: # then needs trimming
        R_padded = R_transformed[-bottom_padding:,...]
        R_shg_padded = R_shg_transformed[-bottom_padding:,...]
    elif bottom_padding == 0:
        R_padded = R_transformed
        R_shg_padded = R_shg_transformed
    
    top_padding = B.shape[0] - R_padded.shape[0]
    if top_padding > 0: # the needs padding
        R_padded = np.concatenate( (R_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
        R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
    elif top_padding < 0: # then needs trimming
        R_padded = R_padded[0:top_padding,...]
        R_shg_padded = R_shg_padded[0:top_padding,...]
    
    output_dir = path.dirname(R_tifs[t])
    
    print('Saving')
    io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
    io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)
    
    
        

    
