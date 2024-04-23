#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:24:22 2022

@author: xies
"""

import numpy as np
from skimage import io, filters, transform, util
from os import path
from tqdm import tqdm
from glob import glob
from re import match
from pystackreg import StackReg
from imageUtils import gaussian_blur_3d

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'

#%% Reading the first ome-tiff file using imread reads entire stack

XX = 1024

def sort_by_day(filename):
    day = match('\d+. Day (\d+\.?5?)',path.split(path.split(filename)[0])[1])
    day = day.groups()[0]
    return float(day)

# Grab all registered B/R tifs
# B_tifs = sorted(glob(path.join(dirname,'*. Day*/ZSeries*/B_reg.tif')),key=sort_by_day)
G_tifs = sorted(glob(path.join(dirname,'*. Day*/G_reg.tif')),key=sort_by_day)
R_shg_tifs = sorted(glob(path.join(dirname,'*. Day*/R_shg_reg.tif')),key=sort_by_day)
R_tifs = sorted(glob(path.join(dirname,'*. Day*/R_reg.tif')),key=sort_by_day)

#% Hand build translation matrices

Tmatrices = dict()
Tmatrices[0] = transform.SimilarityTransform(translation=(-45,65))
Tmatrices[1] = transform.SimilarityTransform(translation=(0,12))
Tmatrices[2] = transform.SimilarityTransform(translation=(5,0))

Tmatrices[3] = transform.SimilarityTransform(translation=(-8,15))
Tmatrices[4] = transform.SimilarityTransform(translation=(5,5))
Tmatrices[5] = transform.SimilarityTransform(translation=(-5,15))
Tmatrices[6] = transform.SimilarityTransform(translation=(60,-118))
Tmatrices[7] = transform.SimilarityTransform(translation=(-13,-30))
Tmatrices[8] = transform.SimilarityTransform(translation=(0,12))
Tmatrices[9] = transform.SimilarityTransform(translation=(0,-10))
Tmatrices[10] = transform.SimilarityTransform(translation=(0,65))
Tmatrices[11] = transform.SimilarityTransform(translation=(39,36))
Tmatrices[12] = transform.SimilarityTransform(translation=(50,-65))
Tmatrices[13] = transform.SimilarityTransform(translation=(0,15))
Tmatrices[14] = transform.SimilarityTransform(translation=(-5,30))

Zshifts = dict()
Zshifts[0] = 14
Zshifts[1] = -9
Zshifts[2] = 5

Zshifts[3] = 2
Zshifts[4] = 8
Zshifts[5] = 12
Zshifts[6] = 9
Zshifts[7] = 4
Zshifts[8] = 4
Zshifts[9] = 2
Zshifts[10] = 20
Zshifts[11] = 9
Zshifts[12] = 4
Zshifts[13] = -8
Zshifts[14] = -8

np.save(arr=(Tmatrices,Zshifts),file=path.join(dirname,'G_R_alignment.npy'))

assert(len(G_tifs) == len(R_tifs))
assert(len(G_tifs) == len(R_shg_tifs))

OVERWRITE = True

 #%% Transform
# 
# for t in tqdm(np.arange(0,14)):
    t = 2
    
    output_dir = path.dirname(R_tifs[t])
    # if path.exists(path.join(output_dir,'R_reg_reg.tif')) and not OVERWRITE:
    #     print(f'Skipping {output_dir}')
    #     continue
    
    s_xy = 0.5
    s_z = 1
    
    G = io.imread(G_tifs[t])
    R = io.imread(R_tifs[t])
    R = gaussian_blur_3d(R,s_xy,s_z)
    R_shg = io.imread(R_shg_tifs[t])
    
    Zshift = Zshifts[t]
    G_zref = 21
    
    # G_ref = G[G_zref,...]
    R_ref = R[G_zref + Zshift,...]
    
    R_transformed = R.copy().astype(float)
    R_shg_transformed = util.img_as_float(R_shg)
    
    T = Tmatrices[t]
    for i, R_slice in enumerate(R):
        R_transformed[i,...] = transform.warp(R_slice.astype(float),T)
        R_shg_transformed[i,...] = transform.warp(R_shg[i,...].astype(float),T)
        
    
    # Z-pad the red + red_shg channel using Imax and Iz
    bottom_padding = Zshift
    if bottom_padding > 0: # the needs padding
        R_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_transformed), axis= 0)
        R_shg_padded = np.concatenate( (np.zeros((bottom_padding,XX,XX)),R_shg_transformed), axis= 0)
    elif bottom_padding < 0: # then needs trimming
        R_padded = R_transformed[-bottom_padding:,...]
        R_shg_padded = R_shg_transformed[-bottom_padding:,...]
    elif bottom_padding == 0:
        R_padded = R_transformed
        R_shg_padded = R_shg_transformed
    
    top_padding = G.shape[0] - R_padded.shape[0]
    if top_padding > 0: # the needs padding
        R_padded = np.concatenate( (R_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
        R_shg_padded = np.concatenate( (R_shg_padded.astype(float), np.zeros((top_padding,XX,XX))), axis= 0)
    elif top_padding < 0: # then needs trimming
        R_padded = R_padded[0:top_padding,...]
        R_shg_padded = R_shg_padded[0:top_padding,...]
    
    
    io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(R_padded/R_padded.max()),check_contrast=False)
    io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(R_shg_padded/R_shg_padded.max()),check_contrast=False)
    
    
    
         
