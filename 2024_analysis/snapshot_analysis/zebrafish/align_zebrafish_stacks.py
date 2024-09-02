#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 18:33:13 2024

@author: xies
"""

import pandas as pd
from os import path
from glob import glob
from natsort.natsort import natsorted
from skimage import io, transform, util, registration
import numpy as np
from tqdm import tqdm

from scipy import ndimage
from pystackreg import StackReg

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%% Load stacks

files = natsorted(glob(path.join(dirname,'stacks','*Position001_t*_ch00*.tif')))
stacks_ch0 = list(map(io.imread,files))
files = natsorted(glob(path.join(dirname,'stacks','*Position001_t*_ch01*.tif')))
stacks_ch1 = list(map(io.imread,files))
files = natsorted(glob(path.join(dirname,'stacks','*Position001_t*_ch02*.tif')))
stacks_ch2 = list(map(io.imread,files))
files = natsorted(glob(path.join(dirname,'stacks','*Position001_t*_ch03*.tif')))
stacks_ch3 = list(map(io.imread,files))

XX = 1024
TT = len(files)

stacks_venus = [x+y for x,y in zip(stacks_ch0,stacks_ch2)]
stacks_mCherry = [x+y for x,y in zip(stacks_ch1,stacks_ch3)]

for i in tqdm(range(len(stacks_venus))):
    stacks_venus[i] = ndimage.gaussian_filter(stacks_venus[i],sigma=[1,1,1])
    stacks_mCherry[i] = ndimage.gaussian_filter(stacks_mCherry[i],sigma=[1,1,1])

#%% Save MIP

for i in tqdm(range(len(stacks_venus))):
    io.imsave(path.join(dirname,f'3d_blur_combined_MIP/venus_MIP_t{i}.tif'),stacks_venus[i].max(axis=0))
    io.imsave(path.join(dirname,f'3d_blur_combined_MIP/mCherry_MIP_t{i}.tif'),stacks_mCherry[i].max(axis=0))

#% Load MIPs as timeseries
# files = natsorted(glob(path.join(dirname,'3d_blur_combined_MIP/venus_MIP*.tif')))
# MIP_venus = np.stack(list(map(io.imread,files)))
# files = natsorted(glob(path.join(dirname,'3d_blur_combined_MIP/mCherry_MIP_t*.tif')))
# MIP_mCherry = np.stack(list(map(io.imread,files)))

MIP_venus = np.stack([x.max(axis=0) for x in stacks_venus])
MIP_mCherry = np.stack([x.max(axis=0) for x in stacks_mCherry])

# Save MIP of z-blurred
# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates/raw_mCherry_MIP.tif'),util.img_as_uint(MIP_mCherry))
# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates//raw_venus_MIP.tif'),util.img_as_uint(MIP_venus))

#%% Stack Reg on MIP timeseries

sr = StackReg(StackReg.RIGID_BODY)

tmats = sr.register_stack(MIP_mCherry, reference='previous')
rough_aligned_mCherry = sr.transform_stack(MIP_mCherry)
rough_aligned_venus = sr.transform_stack(MIP_venus)

# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates/rough_aligned_stacks_mCherry.tif'),
#           util.img_as_uint(rough_aligned_mCherry/rough_aligned_mCherry.max()),check_contrast=False)
# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates/rough_aligned_stacks_venus.tif'),
#           util.img_as_uint(rough_aligned_venus/rough_aligned_venus.max()),check_contrast=False)

initial_tmats = [transform.EuclideanTransform(matrix=x) for x in tmats]
np.savez(path.join(dirname,'initial_tmats.npz'),initial_tmats)

#%% Manually stitch the problematic timepoints
# initial_tmats = np.load(path.join(dirname,'initial_tmats.npz'))['arr_0']

ref_T = 13
target_T = 14

shifted_tmats = [transform.EuclideanTransform(matrix=np.eye(3)) for i in range(target_T)]
reference_image = rough_aligned_venus[ref_T,...]
target_image = rough_aligned_venus[target_T,...]

# shifts = registration.phase_cross_correlation(reference_image, target_image)[0]
T = transform.EuclideanTransform(translation=[100,-40],rotation=np.deg2rad(2))
shifted_mCherry = rough_aligned_mCherry.copy()
shifted_venus = rough_aligned_venus.copy()

for t in tqdm(np.arange(target_T,TT)):
    
    warped = transform.warp(rough_aligned_mCherry[t,...],T)
    shifted_mCherry[t,...] = warped
    warped = transform.warp(rough_aligned_venus[t,...],T)
    shifted_venus[t,...] = warped
    shifted_tmats.append(T)

io.imsave(path.join(dirname,'shifted_mCherry_MIP.tif'),util.img_as_uint(shifted_mCherry/shifted_mCherry.max()))
io.imsave(path.join(dirname,'shifted_venus_MIP.tif'),util.img_as_uint(shifted_venus/shifted_venus.max()))
np.savez(path.join(dirname,'shifted_tmats.npz'),shifted_tmats)

#%% Rerun SR to refine final XY transformation and then resave

# shifted_tmats = np.load(path.join(dirname,'shifted_tmats.npz'))['arr_0']
sr = StackReg(StackReg.RIGID_BODY)
tmats = sr.register_stack(shifted_venus,reference='previous')
refined_venus = sr.transform_stack(shifted_venus)
refined_mCherry = sr.transform_stack(shifted_venus)

# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates/refined_mCherry_MIP.tif'),util.img_as_uint(refined_mCherry/refined_mCherry.max()))
# io.imsave(path.join(dirname,'3d_blurred_MIP_alignment_intermediates/refined_venus_MIP.tif'),util.img_as_uint(refined_venus/refined_venus.max()))

refined_tmats = [transform.EuclideanTransform(x) for x in tmats]

np.savez(path.join(dirname,'refined_tmats.npz'),refined_tmats)

#%% Use the final TMATs to transform the stacks

initial_tmats = np.load(path.join(dirname,'initial_tmats.npz'))['arr_0']
shifted_tmats = np.load(path.join(dirname,'shifted_tmats.npz'))['arr_0']
# refined_tmats = np.load(path.join(dirname,'refined_tmats.npz'))['arr_0']

tmat_tuples = zip(initial_tmats,shifted_tmats)
total_tmats = [transform.EuclideanTransform(x)
               + transform.EuclideanTransform(y) for x,y in tmat_tuples]

mCherry_XY_transformed = []
venus_XY_transformed = []

for t in tqdm(range(len(stacks_mCherry))):
    
    T = transform.EuclideanTransform(matrix=total_tmats[t])
    T2 = transform.EuclideanTransform(matrix=shifted_tmats[t])
    _stack_mCh = np.zeros_like(stacks_mCherry[t]).astype(float)
    _stack_ven = np.zeros_like(stacks_venus[t]).astype(float)
    
    for z in range(stacks_mCherry[t].shape[0]):
        im = transform.warp(stacks_mCherry[t][z,...],T)
        _stack_mCh[z,...] = im
        im = transform.warp(stacks_venus[t][z,...],T)
        _stack_ven[z,...] = im
        
    mCherry_XY_transformed.append(_stack_mCh)
    venus_XY_transformed.append(_stack_ven)

#%% Align z-slices

from twophotonUtils import z_align_ragged_timecourse

# Use manual z-alignment
same_Zs = pd.read_csv(path.join(dirname,'same_Zs.csv'),index_col=0)['slice']

aligned_stack_mch = z_align_ragged_timecourse(mCherry_XY_transformed,same_Zs)
aligned_stack_venus = z_align_ragged_timecourse(venus_XY_transformed,same_Zs)

#%% Save final alignments

for t in tqdm(np.arange(0,TT)):
    im = aligned_stack_mch[t,7:45]
    io.imsave(path.join(dirname,f'aligned_stacks/aligned_stack_mch_t{t:02d}.tif'),util.img_as_uint(im/im.max()))
    im = aligned_stack_venus[t,7:45]
    io.imsave(path.join(dirname,f'aligned_stacks/aligned_stack_venus_t{t:02d}.tif'),util.img_as_uint(im/im.max()))

#%%



