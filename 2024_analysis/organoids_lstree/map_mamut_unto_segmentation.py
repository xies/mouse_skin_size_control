#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:58:19 2024

@author: xies
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from natsort import natsort
from glob import glob
from skimage import io

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
dx = 0.26
dz = 2
XX = 396
YY = 404
ZZ = 42

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)

# Deaggregate into list of unique tracks
tracks = [t for _,t in df.groupby('TrackID')]

#%% Go through each track and 'filter' the segmentation

filenames = natsort.natsorted(glob(path.join(dirname,'manual_segmentation/*.tif')))
manual_segmentations = np.stack(list(map(io.imread,filenames)))

filt_seg = np.zeros((34,ZZ,YY,XX),dtype=int)

#%%
track = tracks[1]
track['X-pixel'] = np.round(track['X']/dx).astype(int)
track['Y-pixel'] = np.round(track['Y']/dx).astype(int)
track['Z-pixel'] = np.round(track['X']/dz).astype(int)

for idx,row in track.iterrows():
    t = int(row['FRAME'])
    x,y,z = row[['X-pixel','Y-pixel','Z-pixel']]
    if t < 35:
        label = manual_segmentations[int(t),z,y,x]
        if label > 0:
            stop