#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:04:52 2023

@author: xies
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:29:22 2022

@author: xies
"""

import numpy as np
from skimage import io, transform, util
from os import path
from glob import glob
from pystackreg import StackReg
from tqdm import tqdm

from twophotonUtils import sort_by_prefix

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1'

#%% Reading the first ome-tiff file using imread reads entire stack

# Grab all registered B/R tifs
B_tifs = sorted(glob(path.join(dirname,'*. */' + 'B_reg.tif')), key = sort_by_prefix)
G_tifs = sorted(glob(path.join(dirname,'*. */' + 'G_reg.tif')), key = sort_by_prefix)

#%% Correlate each R_shg timepoint with subsequent timepoint (Nope, using first time point instead)
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

XX = 1024

ref_T = 0
target_T = 3

###
B_ref = io.imread(B_tifs[ref_T])
B_target = io.imread(B_tifs[target_T])
G_target = io.imread(G_tifs[target_T])

# Grab the manually determined reference slice
Imax_ref = 62
ref_img = B_ref[Imax_ref,...]
ref_img = ref_img / ref_img.max()
Z_ref = B_ref.shape[0]

output_dir = path.split(path.dirname(B_tifs[target_T]))[0]

# Grab the target slice
Imax_target = 72
target_img = B_target[Imax_target,...]
target_img = target_img / target_img.max()

print('\n Starting stackreg')
# Use StackReg to 'align' the two z slices
sr = StackReg(StackReg.RIGID_BODY)
T = sr.register(ref_img,target_img)

T = transform.SimilarityTransform(matrix=T)
T1 = transform.SimilarityTransform(translation=[30,-30],rotation=np.deg2rad(2))
T = T+T1

print('Applying transformation matrices')
# Apply transformation matrix to each stacks
B_transformed = np.zeros_like(B_target)
G_transformed = np.zeros_like(G_target)

for i, B_slice in enumerate(B_target):
    B_transformed[i,...] = transform.warp(B_slice.astype(float),T)
    G_transformed[i,...] = transform.warp(G_target[i,...].astype(float),T)

# Z-pad the time point in reference to t - 1
Z_target = B_target.shape[0]

print('Padding')
top_padding = Imax_ref - Imax_target
if top_padding > 0: # the needs padding
    G_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),G_transformed), axis= 0)
    B_padded = np.concatenate( (np.zeros((top_padding,XX,XX)),B_transformed), axis= 0)
    
elif top_padding < 0: # then needs trimming 
    G_padded = G_transformed[-top_padding:,...]
    B_padded = B_transformed[-top_padding:,...]
    
elif top_padding == 0:
    # R_padded = R
    G_padded = G_transformed
    B_padded = B_transformed
    # R_shg_padded = R_shg_target
    
delta_ref = Z_ref - Imax_ref
delta_target = Z_target - Imax_target
bottom_padding = delta_ref - delta_target
if bottom_padding > 0: # the needs padding
    G_padded = np.concatenate( (G_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    B_padded = np.concatenate( (B_padded.astype(float), np.zeros((bottom_padding,XX,XX))), axis= 0)
    
elif bottom_padding < 0: # then needs trimming
    G_padded = G_padded[0:bottom_padding,...]
    B_padded = B_padded[0:bottom_padding,...]
    
print('Saving')
output_dir = path.dirname(B_tifs[target_T])
io.imsave(path.join(output_dir,'B_align.tif'),util.img_as_uint(B_padded/B_padded.max()))
io.imsave(path.join(output_dir,'G_align.tif'),util.img_as_uint(G_padded/G_padded.max()))

