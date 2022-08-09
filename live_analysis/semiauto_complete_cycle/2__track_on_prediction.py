#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:41:36 2021

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

import pickle as pkl

dirnames = {}
dirname = '/Users/xies//OneDrive - Stanford/Skin/06-25-2022/M6 RBKO/R1/manual_track'

# dx = 0.2920097
dx = 1

#%% Load data

# Load preliminary tracks
with open(path.join(dirname,'MaMuT','complete_cycles.pkl'),'rb') as file:
    tracks = pkl.load(file)

# Load prediction by stardist
# filenames = [path.join(dirname,f'reg/prediction/z_reg_t{t}_chan2.tif') for t in range(19)]
# seg = np.array([ io.imread(f) for f in filenames ])

# Load prediction by stardist
filenames = [path.join(dirname,f'reg/prediction/z_reg_t{t}_chan2.tif') for t in range(19)]
seg = np.array([ io.imread(f) for f in filenames ])

# with open(path.join(dirname,'MaMuT','complete_cycles_seg.pkl'),'rb') as file:
#     cells = pkl.load(file)

# Load final tracks
# with open(path.join(dirname,'manual_track','complete_cycles_seg.pkl'),'rb') as file:
#     tracks = pkl.load(file)

#%% Use tracks and extract segmentation; generate a filtered segmentation image
# where only tracked spots are shown + put 3D markers on un-segmented spots

radius = 5

[T,Z,X,Y] = seg.shape
seg_filt = np.zeros_like(seg,dtype=np.int8)

# Filter segmentation based on complete tracks
trackID = 1
for track in tracks:
    
    track['TrackID'] = trackID
    trackID += 1
    for idx,spot in track.iterrows():
        x = int(spot['X']/dx)
        y = int(spot['Y']/dx)
        z = int(np.round(spot['Z']))
        t = int(spot['Frame'])
        
        this_seg = seg[t,...]
        this_seg_filt = seg_filt[t,...]
        
        label = this_seg[z,y,x]
        
        track.at[idx,'Segmentation'] = label
        
        
        if label > 0:
            # filter¸segmentation image to only include tracked spots
            this_seg_filt[this_seg == label] = trackID
        else:
            # Create a 'ball' around spots missing
            # print(f'Time {t} -- {i}: {ID}')
            y_low = max(0,y - radius); y_high = min(Y,y + radius)
            x_low = max(0,x - radius); x_high = min(X,x + radius)
            z_low = max(0,z - radius); z_high = min(Z,z + radius)
            this_seg_filt[z_low:z_high, y_low:y_high, x_low:x_high] = trackID


io.imsave(path.join('/Users/xies/Desktop/filtered_segmentation.tif'),
      seg_filt.astype(np.int8))

