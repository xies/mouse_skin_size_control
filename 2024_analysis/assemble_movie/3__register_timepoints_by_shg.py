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

from twophotonUtils import parse_unaligned_channels, z_translate_and_pad

# dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Right ear/Post Ethanol/R3'
# dirname = '/Volumes/T7/01-13-2023 Ablation K14Cre H2B FUCCI/Black unclipped less leaky DOB 06-30-2023/R2'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/Old mice/01-24-2024 12month old mice/F1 DOB 12-18-2022/R1'
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Two photon/NMS/Old mice/04-30-2024 16month old mice/M3 DOB 12-27-2022/R1/'

filelist = parse_unaligned_channels(dirname,folder_str='*. Day*/')

#%%

XX = 1024
TT = len(filelist)

ALIGN_TO_ALIGNED = False

XY_reg = True
APPLY_XY = True
APPLY_PAD = True

ref_T = 0

manual_Ztarget = {}

z_pos_in_original = {}
XY_matrices = {}
if path.exists(path.join(dirname,'alignment_information.pkl')):
    with open(path.join(dirname,'alignment_information.pkl'),'rb') as f:
        [z_pos_in_original,XY_matrices,Imax_ref] = pkl.load(f)

# Find the slice with maximum mean value in R_shg channel
if ALIGN_TO_ALIGNED:
    ref_subdir = path.dirname(filelist.loc[ref_T,'R_shg'])
    R_shg_ref = io.imread(path.join(ref_subdir,'R_shg_align.tif'))
else:
    R_shg_ref = io.imread( filelist.loc[ref_T,'R_shg'] )

Z_ref = R_shg_ref.shape[0]
Imax_ref = R_shg_ref.std(axis=2).std(axis=1).argmax() # Find max contrast slice
Imax_ref = 33
ref_img = R_shg_ref[Imax_ref,...]
print(f'Reference z-slice: {Imax_ref}')

# variables to save:
z_pos_in_original[ref_T] = Imax_ref

#%% Z-target and XY transform
# Correlate each R_shg timepoint with first time point
# R_shg is best channel to use bc it only has signal in the collagen layer.
# Therefore it's easy to identify which z-stack is most useful.

OVERWRITE = True
for t in tqdm( [3] ): # 0-indexed

    ref_T = 1
    if t == ref_T:
        continue

    output_dir = path.split(path.dirname(filelist.loc[t,'R']))[0]
    if not OVERWRITE and path.exists(path.join(path.dirname(filelist.loc[t,'G']),'G_align.tif')):
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
            # T = T + transform.SimilarityTransform(translation=[30,75],rotation=np.deg2rad(-7))

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
        dz = Imax_ref - Imax_target
        # Tz = transform.EuclideanTransform(translation=[0,-dz])
        R_padded = z_translate_and_pad(R_shg_ref,R_transformed,Imax_ref,Imax_target)
        B_padded = z_translate_and_pad(R_shg_ref,B_transformed,Imax_ref,Imax_target)
        G_padded = z_translate_and_pad(R_shg_ref,G_transformed,Imax_ref,Imax_target)
        R_shg_padded = z_translate_and_pad(R_shg_ref,R_shg_transformed,Imax_ref,Imax_target)

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

#%%
print('DONE')

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
