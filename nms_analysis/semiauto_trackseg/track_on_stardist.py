#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:41:36 2021

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io
import seaborn as sb
from os import path

import pickle as pkl

# Avoid parsing XML
# import xml.etree.ElementTree as ET

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%% Load data

with open(path.join(dirname,'complete_cycles.pkl'),'rb') as file:
    tracks = pkl.load(file)

# Load prediction by stardist
# seg = io.imread(path.join(dirname,'stardist/prediction.tif'))

#%% Use tracks and extract segmentation

for track in tracks:
    
    for idx,spot in track.iterrows():
        
        x = int(spot['X'])
        y = int(spot['Y'])
        z = int(spot['Z'])
        t = int(spot['Frame'])
        
        label = seg[t,z,y,x]
        
        track.at[idx,'Segmentation'] = label
    

#%% Make measurements from segmentation

for track in tracks:
    for idx,spot in track.iterrows():
        
        segID = spot['Segmentation']
        if segID > 0:
            t = int(spot['Frame'])
            volume = (seg[t,...] == segID).sum()
        else:
            volume = np.nan
            
        track.at[idx,'Volume'] = volume
    # Pad out last time point for plotting ease
    track.loc['padding',:] = np.nan
ts = pd.concat(tracks)


with open(path.join(dirname,'complete_cycles_seg.pkl'),'wb') as file:
    pkl.dump(tracks,file)

#%% Pull out the segmentation that are used in tracks and write new image stack

track_seg = np.zeros(seg.shape)
for track in tracks:
    for _,spot in track.iterrows():
        
        segID = spot['Segmentation']
        if segID > 0:
            t = int(spot['Frame'])
            track_seg[t, seg[t,...] == segID] = spot['TrackID']

io.imsave(path.join(dirname,'tracked.tif'),track_seg.astype(np.int16))
