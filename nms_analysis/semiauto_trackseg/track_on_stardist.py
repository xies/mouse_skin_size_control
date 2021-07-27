#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:41:36 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, morphology
from scipy.ndimage import convolve
import seaborn as sb
from os import path
from glob import glob

import pickle as pkl

# Avoid parsing XML
# import xml.etree.ElementTree as ET

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%% Load data

with open(path.join(dirname,'tracking','complete_cycles.pkl'),'rb') as file:
    tracks = pkl.load(file)

# Load prediction by stardist
seg = io.imread(path.join(dirname,'stardist/prediction1.tif'))

#%% Use tracks and extract segmentation; generate a filtered segmentation image
# where only tracked spots are shown + put 3D markers on un-segmented spots

radius = 5

[T,Z,X,Y] = seg.shape
seg_filt = np.zeros_like(seg,dtype=np.int16)

# Filter segmentation based on complete tracks
trackID = 1
for track in tracks:
    
    track['TrackID'] = trackID
    trackID += 1
    for idx,spot in track.iterrows():
        x = int(spot['X'])
        y = int(spot['Y'])
        z = int(spot['Z'])
        t = int(spot['Frame'])
        
        this_seg = seg[t,...]
        this_seg_filt = seg_filt[t,...]
        
        label = this_seg[z,y,x]
        
        track.at[idx,'Segmentation'] = label
        
        
        if label > 0:
            # filter¸segmentation image to only include tracked spots
            this_seg_filt[this_seg == label] = spot.ID
        else:
            # Create a 'ball' around spots missing
            y_low = max(0,y - radius); y_high = min(Y,y + radius)
            x_low = max(0,x - radius); x_high = min(X,x + radius)
            z_low = max(0,z - radius); z_high = min(Z,z + radius)
            this_seg_filt[z_low:z_high, y_low:y_high, x_low:x_high] = spot.ID

#Re-index each segmentation from 1 onwards (labkit tries to generate labels in between integers if labels are sparse
for t in range(T):
    this_seg_filt = seg_filt[t,...]
    # Reindex from 1
    uniqueIDs = np.unique(this_seg_filt)
    for i,ID in enumerate(uniqueIDs):
        this_seg_filt[this_seg_filt == ID] = i
        
        print(f'Time {t} -- {i}: {ID}')
        
    # io.imsave(path.join('/Users/xies/Desktop/',f'seg_filt_t{t}.tif'),this_seg_filt)

#%% Load and collate manual track+segmentations ()

# Dictionary of manual segmentation (there should be no first or last time point)
manual_segs = np.zeros((17,69,1024,1024),dtype=np.int16)
for t in range(T):
    filename = path.join(dirname,'tracking/tracked_segmentation/manual/tif',f't{t}.tif')
    if path.exists(filename):
    
        manual_segs[t,...] = io.imread(filename)

#%%

corrected_segs = np.zeros_like(manual_segs)
for trackID,track in enumerate(tracks):
    for i,spot in track.iterrows():
        x = int(spot['X'])
        y = int(spot['Y'])
        z = int(spot['Z'])
        t = int(spot['Frame'])
        
        this_seg = manual_segs[t,...]
        label = this_seg[z,y,x]
        if label == 0:
            print(f'Error at t = {t}, ID = {spot.ID}, segID = {label}')
        else:
            corrected_segs[t,this_seg == label] = trackID+1
            track.at[i, 'CorrID'] = trackID+1


# io.imsave('/Users/xies/Desktop/manual_seg.tif', corrected_segs.astype(np.int8))

#%% Make measurements from segmentation

for track in tracks:
    for idx,spot in track.iterrows():
        
        segID = spot['CorrID']
        if segID > 0:
            t = int(spot['Frame'])
            volume = (corrected_segs[t,...] == segID).sum()
            
        else:
            volume = np.nan
            
        track.at[idx,'Volume'] = volume
        # Pad out last time point for plotting ease
        track.loc['padding',:] = np.nan
        

# Calculate time since birth
for track in tracks:
    track['Time'] =( track['Frame'] - track.iloc[0]['Frame'])*12
        
ts = pd.concat(tracks)

with open(path.join(dirname,'complete_cycles_seg.pkl'),'wb') as file:
    pkl.dump(tracks,file)

