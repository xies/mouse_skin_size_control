#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 19:09:05 2025

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import seaborn as sb
from os import path

import pickle as pkl
from tqdm import tqdm

#%% Export the coordinates of the completed cell cycles (as pickle)

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'
dz = 1; dx = 0.25

with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)

all_tracks.append(tracks)

#% Load nuclear seg - basal v. suprabasal
filenames = natsort.natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t*.tif')))
basal_segs = np.stack( list(map(io.imread, filenames) ) )

filenames = natsort.natsorted(glob(path.join(dirname,'3d_nuc_seg_supra/cellpose_manual/t*.tif')))
suprabasal_segs = np.stack( list(map(io.imread, filenames) ) )

#%%

def get_cube_fill_as_slice(im_shape,centroid,side_length=3):
    assert( len(im_shape) == len(centroid) )
    
    lower_bounds = centroid - side_length
    upper_bounds = centroid + side_length

    lower_bounds = np.fmax(lower_bounds,np.zeros(len(centroid)))
    upper_bounds = np.fmin(upper_bounds,im_shape)
    
    slice_tuple = (slice(int(lower_bounds[0]),int(upper_bounds[0])),
                   slice(int(lower_bounds[1]),int(upper_bounds[1])),
                   slice(int(lower_bounds[2]),int(upper_bounds[2])) )
    
    return slice_tuple


tracked_seg = np.zeros_like(basal_segs)

for track in tqdm(tracks):

    for idx,spot in track.iterrows():
        frame = int(float(spot['Frame']))
        Z,Y,X = spot[['Z','Y','X']].astype(float)
        Z = int(Z);
        Y = int(Y/dx); X = int(X/dx)
        
        if spot['Cell type'] == 'Basal':
            label = basal_segs[frame,Z,Y,X]
            if label > 0:
                tracked_seg[frame,basal_segs[frame,...] == label] = spot['TrackID']
        elif spot['Cell type'] == 'Suprabasal':
            label = suprabasal_segs[frame,Z,Y,X]
            if label > 0:
                tracked_seg[frame,suprabasal_segs[frame,...] == label] = spot['TrackID']
            else:
                print(f'{frame}, {label}')
                sli = get_cube_fill_as_slice(tracked_seg[frame,...].shape,
                                             np.array([Z,Y,X]))
                tracked_seg[frame,sli[0],sli[1],sli[2]] = spot['TrackID']

io.imsave('/Users/xies/Desktop/tracked_seg.tif',tracked_seg)
    


