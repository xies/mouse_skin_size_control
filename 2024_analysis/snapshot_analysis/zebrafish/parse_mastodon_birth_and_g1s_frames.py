#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:07:48 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

_spots = pd.read_csv(path.join(dirname,'Position001_Mastodon/birth/spots.csv'),skiprows=[1,2])
_links = pd.read_csv(path.join(dirname,'Position001_Mastodon/birth/linkage.csv'),skiprows=[1,2])
_links = _links.rename(columns={'Link target IDs':'SourceID','Link target IDs.1':'TargetID'})

for l in _links.iterrows():
    print(l)
    


#%% Load parsed tracks, previous manual segtrack, additional segonly

MANUAL = True

# Load preliminary tracks
with open(path.join(dirname,'MaMuT/Ablation','dense_tracks.pkl'),'rb') as file:
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
for t in range(7):
    segonly.append(io.imread(path.join(dirname,f'cellpose_G_clahe_blur/t{t}_3d_nuc/t{t}_masks.tif')))
segonly = np.stack(segonly)

if MANUAL:
    segtrack = io.imread(path.join(dirname,'manual_tracking/Ablation_R20_Ablation.tif'))
else:
    segtrack = np.zeros_like(segonly,dtype=np.int16)

#%% Use tracks and extract segmentation; generate a filtered segmentation image
# where only tracked spots are shown + put 3D markers on un-segmented spots

radius = 5

[T,Z,X,Y] = segtrack.shape


# Filter segmentation based on complete tracksvi
trackID = 0
for track in tqdm(tracks):
    
    track['TrackID'] = trackID
    trackID += 1
    for idx,spot in track.iterrows():
        x = int(spot['X']/dx)
        y = int(spot['Y']/dx)
        z = int(np.round(spot['Z']))
        t = int(spot['Frame'])
        
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
        segtrack[t,...] = this_segtrack

io.imsave(path.join('/Users/xies/Desktop/filtered_segmentation.tif'),
      segtrack.astype(np.uint16))




