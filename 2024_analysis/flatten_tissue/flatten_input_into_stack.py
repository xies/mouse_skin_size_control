#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:36:03 2022

@author: xies
"""


import numpy as np
from skimage import io
from scipy.ndimage import affine_transform
from glob import glob
from natsort import natsorted
from os import path
from tqdm import tqdm

from re import match

# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/YAP-KO ablation/04-07-2-25 YAP-KO ablation/F1 YT-fl K14Cre DOB 02-10-2025/Left ear 4OHT day 3/R1 near distal edge/'

def sort_by_timestamp(filename):
    t = match('t(\d+).tif',filename).groups[0]
    return int(t)
    

#%% Load a heightmap and flatten the given z-stack

TOP_OFFSET = 30
BOTTOM_OFFSET = -10

<<<<<<< Updated upstream
<<<<<<< Updated upstream

filenames = natsorted(glob(path.join(dirname,'*.*/G_reg.tif')))
T = len(filenames)

XY_mats = np.load(path.join(dirname,'alignments/2D_affine_matrices.npy'))

=======
filenames = natsorted(glob(path.join(dirname,'*.*/G_reg.tif')))
T = len(filenames)

XY_mats = np.load(path.join(dirname,'alignments/2D_affine_matrices.npy'))

>>>>>>> Stashed changes
=======
filenames = natsorted(glob(path.join(dirname,'*.*/G_reg.tif')))
T = len(filenames)

XY_mats = np.load(path.join(dirname,'alignments/2D_affine_matrices.npy'))

>>>>>>> Stashed changes
for t in tqdm(range(T)):
    
    if t == 0:
        continue
    
    im = io.imread(filenames[t])
    Z,XX,_ = im.shape
    
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    M = np.linalg.inv(XY_mats[t,...])
    rot = M[:2,:2]
    transl = M[:2,2]
    
    output_dir = path.join(dirname,'Image flattening/Flat stacks')
    
    flat = np.zeros((TOP_OFFSET-BOTTOM_OFFSET,XX,XX))
    Iz_top = heightmap + TOP_OFFSET
    Iz_bottom = heightmap + BOTTOM_OFFSET
    
    for x in range(XX):
        for y in range(XX):
            
            flat_indices = np.arange(0,TOP_OFFSET-BOTTOM_OFFSET)
                
            z_coords = np.arange(Iz_bottom[y,x],Iz_top[y,x])
            # sanitize for out-of-bounds
            # z_coords[z_coords < 0] = 0
            # z_coords[z_coords >= Z] = Z-1
            I = (z_coords > 0) & (z_coords < Z)
            
            flat[flat_indices[I],y,x] = im[z_coords[I],y,x]
    
    for z in range(flat.shape[0]):
        flat[z,...] = affine_transform(flat[z,...],rot,transl)
    
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16))
    
