#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:18:41 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, morphology, util, measure
from scipy.ndimage import convolve
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

dirname = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Post tam/R1/'

# dx = 0.2920097
dx = 1

#%% Load parsed tracks, previous manual segtrack, additional segonly

MANUAL = False

# Load preliminary tracks
with open(path.join(dirname,'MaMuT','complete_cycles.pkl'),'rb') as file:
    tracks = pkl.load(file)

# Convert prediction by cellpose (.npz) into *_masks.tif
# see: https://github.com/MouseLand/cellpose/blob/main/docs/outputs.rst

# filenames = glob(path.join(dirname,f'im_seq/t*.npy'))
# for f in filenames:
#     out_name = path.splitext(f)[0] + '_seg.tif'
#     if path.exists(out_name):
#         continue
#     data = np.load(f,allow_pickle=True).item()
#     seg = data['masks']
#     io.imsave(out_name, seg)
#     io.imsave(path.splitext(f)[0] + '_prob.tif',data['flows'][3])

segonly = []
for t in range(17):
    segonly.append(io.imread(path.join(dirname,f'cellpose_B_clahe/t{t}_3d_nuc/t{t}_masks.tif')))
segonly = np.stack(segonly)

if MANUAL:
    segtrack = io.imread(path.join(dirname,'manual_tracking/Ablation_R4_Nonablation.tif'))
else:
    segtrack = np.zeros_like(segonly,dtype=np.int16)

#%%

# Load dfield files
dfields = {}
for t in range(1,15):
    if t == 9:
        continue
    dfield_file = glob(path.join(dirname,f'{t+10}. Day */dfield.tif'))
    
    assert(len(dfield_file) == 1)
    dfields[t] = io.imread(path.join(dfield_file[0]))
    
#%% Use tracks and extract segmentation; generate a filtered segmentation image
# where only tracked spots are shown + put 3D markers on un-segmented spots

radius = 5

[T,Z,X,Y] = segtrack.shape

# Filter segmentation based on complete tracks

trackID = 0
for track in tqdm(tracks):
    
    track['TrackID'] = trackID
    trackID += 1
    for idx,spot in track.iterrows():
        x = int(spot['X']/dx)
        y = int(spot['Y']/dx)
        z = int(np.round(spot['Z']))
        t = int(spot['Frame'])
        
        if t == 9: # t=9 is warped version of 19.Day9.5, which will be replaced by frame t=10 which is unwarped
            continue
        if t > 9:
            t = t - 1 # these are the wrong frame number since t=9 is a dummy frame
            
        if t != 0 and t != 9:
            
            dfield = dfields[t]
            warped_coords = np.array([z,y,x]).astype(int)
            # bound the warped coords so they are inside the dim of dfield
            ZZ = dfield.shape[0]
            warped_coords[0] = min([warped_coords[0],ZZ-1])
            unwarped_coords = np.round(warped_coords + dfield[tuple(warped_coords)][::-1])
            
            z = int(unwarped_coords[0])
            y = int(unwarped_coords[1])
            x = int(unwarped_coords[2])
            
        # If already loading manual, check that manual has tracks already at this frame
        # if so, skip this timepoint
        this_segtrack = segtrack[t,...]
        
        if MANUAL:
            label = this_segtrack[z,y,x]
            if label > 0: # label exits in MANUAL image
                
                this_segtrack[this_segtrack == label] = trackID
                track.at[idx,'Segmentation'] = label
                track.at[idx,'New'] = False
                continue
        
        this_seg = segonly[t,...]
        label = this_seg[z,y,x]
        
        track.at[idx,'Segmentation'] = label
        track.at[idx,'New'] = True
        
        if label > 0:
            # filter¸segmentation image to only include tracked spots
            this_segtrack[this_seg == label] = trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            # print(f'Time {t} -- {i}: {ID}')
            y_low = max(0,y - radius); y_high = min(Y,y + radius)
            x_low = max(0,x - radius); x_high = min(X,x + radius)
            z_low = max(0,z - radius); z_high = min(Z,z + radius)
            this_segtrack[z_low:z_high, y_low:y_high, x_low:x_high] = trackID
            if t == 4:
                stop
        segtrack[t,...] = this_segtrack

io.imsave(path.join('/Users/xies/Desktop/filtered_segmentation.tif'),
      segtrack.astype(np.uint16))




