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
from re import findall
from imageUtils import fill_in_cube

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)

# Deaggregate into list of unique tracks
tracks = [t for _,t in df.groupby('TrackID')]

#%% Go through each track and 'filter' the segmentation

manifest = pd.DataFrame({'filename':natsort.natsorted(glob(path.join(dirname,'manual_segmentation/*.tif')))})
pattern = r'T(\d{4}).tif'
manifest['timestr'] = manifest['filename'].apply(lambda x: findall(pattern, x)[0])
manifest['T'] = manifest['timestr'].astype(int) - 1
# Dict of manual segmentations keyed on the actual time frame (in 0-index)
manual_segmentations = dict(map(
    lambda i,j: (i,j) , manifest['T'].values,  map(io.imread,manifest['filename'].values)) )

ZZ,YY,XX = manual_segmentations[0].shape

#%%

# trackID = 0
filt_seg = np.zeros((40,ZZ,YY,XX),dtype=int)

for track in tqdm(tracks):
    trackID = track.iloc[0]['TrackID']
    track['X-pixel'] = np.round(track['X']/dx).astype(int)
    track['Y-pixel'] = np.round(track['Y']/dx).astype(int)
    track['Z-pixel'] = np.round(track['Z']/dz).astype(int)
    
    for idx,row in track.iterrows():
        
        t = int(row['FRAME'])
        x,y,z = row[['X-pixel','Y-pixel','Z-pixel']]
        if t < 40:
            this_seg = manual_segmentations[int(t)]
            label = this_seg[z,y,x]
            if label > 0:
                mask = this_seg == label
                filt_seg[t,mask] = trackID
            else:
                filt_seg[t,...] = fill_in_cube(filt_seg[t,...],
                                               [z,y,x], trackID)
                print(f'Fill in at t = {t} and trackID = {trackID}')

io.imsave(path.join(dirname,'manual_cellcycle_annotations/filtered_segs.tif'), filt_seg)




