#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:04:52 2023

@author: xies
"""


import numpy as np
from skimage import io, transform, util
from os import path
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm

from twophotonUtils import return_prefix

dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Post tam/R2'

#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
B_tifs = sorted(glob(path.join(dirname,'14. */' + 'B_reg.tif')), key = return_prefix)
R_tifs = sorted(glob(path.join(dirname,'14. */' + 'R_reg.tif')), key = return_prefix)
R_shg_tifs = sorted(glob(path.join(dirname,'14. */' + 'R_shg_reg.tif')), key = return_prefix)

#%% Correlate each R_shg timepoint with subsequent timepoint (Nope, using first time point instead)
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024

###
ref = io.imread(B_tifs[0])
target = io.imread(R_tifs[0])
other_target = io.imread(R_shg_tifs[0])

# Grab the manually determined reference slice
Imax_ref = 63
ref_img = ref[Imax_ref,...]
ref_img = ref_img / ref_img.max()
Z_ref = ref.shape[0]

output_dir = path.split(path.dirname(B_tifs[0]))[0]

# Grab the target slice
Imax_target = 56
target_img = target[Imax_target,...]
target_img = target_img / target_img.max()

print('\n Starting stackreg')
# Use StackReg to 'align' the two z slices
sr = StackReg(StackReg.RIGID_BODY)
T = sr.register(ref_img,target_img)

T = transform.SimilarityTransform(matrix=T)
# T1 = transform.SimilarityTransform(translation=[-20,10],rotation=np.deg2rad(0))
# T = T+T1

print('Applying transformation matrices')
# Apply transformation matrix to each stacks
target_transformed = np.zeros_like(target)
other_target_transformed = np.zeros_like(other_target)

for i, R_slice in enumerate(target):
    target_transformed[i,...] = transform.warp(R_slice.astype(float),T)
    other_target_transformed[i,...] = transform.warp(other_target[i,...].astype(float),T)

# Z-pad the time point in reference to t - 1
Z_target = target.shape[0]

print('Padding')
top_padding = Imax_ref - Imax_target
if top_padding > 0: # the needs padding
    target_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),target_transformed), axis= 0)
    other_target_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),other_target_transformed), axis= 0)
    
elif top_padding < 0: # then needs trimming 
    target_padded = target_transformed[-top_padding:,...]
    other_target_padded = other_target_transformed[-top_padding:,...]
    
elif top_padding == 0:
    # R_padded = R
    target_padded = target_transformed
    other_target_padded = other_target_transformed
    # R_shg_padded = R_shg_target
    
delta_ref = Z_ref - Imax_ref
delta_target = Z_target - Imax_target
bottom_padding = delta_ref - delta_target
if bottom_padding > 0: # the needs padding
    target_padded = np.concatenate( (target_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    other_target_padded = np.concatenate( (other_target_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    
elif bottom_padding < 0: # then needs trimming
    target_padded = target_padded[0:bottom_padding,...]
    other_target_padded = other_target_padded[0:bottom_padding,...]
    
print('Saving')
output_dir = path.dirname(R_tifs[0])
io.imsave(path.join(output_dir,'R_reg_reg.tif'),util.img_as_uint(target_padded/target_padded.max()))
io.imsave(path.join(output_dir,'R_shg_reg_reg.tif'),util.img_as_uint(other_target_padded/other_target_padded.max()))

