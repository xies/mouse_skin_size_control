#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 19:09:05 2025

@author: xies
"""

import numpy as np
import pandas as pd
# import matplotlib.pylab as plt
from skimage import io, measure
import tifffile

# import seaborn as sb
from os import path
from glob import glob
from natsort import natsorted
from imageUtils import most_likely_label

import pickle as pkl
# from tqdm import tqdm

#%% Load segmentations

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

dz = 1; dx = 0.25
with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)

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

#% Load nuclear seg - basal v. suprabasal
filenames = natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t*_basal.tif')))
basal_segs = np.stack( list(map(io.imread, filenames) ) )

filenames = natsorted(glob(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t*_supra.tif')))
suprabasal_segs = np.stack( list(map(io.imread, filenames)) )

#% Load cyto segs - basal only
filenames = natsorted(glob(path.join(dirname,'3d_cyto_seg/3d_cyto_manual/t*_cleaned.tif')))
cyto_segs = np.stack( list(map(io.imread,filenames) ) )

filenames = natsorted(glob(path.join(dirname,'3d_cyto_seg_supra/3d_cyto_supra_raw/t*.tif')))
cyto_supra = np.stack( list(map(io.imread,filenames) ) )

#% Track Mastodon onto segmentation

tracked_nuc = np.zeros_like(basal_segs)
tracked_cyto = np.zeros_like(cyto_segs)

for track in tracks:
    
    for idx,spot in track.iterrows():
        
        frame = int(float(spot['Frame']))
        Z,Y,X = spot[['Z','Y','X']].astype(float)
        Z = int(Z);
        Y = int(Y/dx); X = int(X/dx)
        # print(f'{frame}, {spot.TrackID}')
        
        if spot['Cell type'] == 'Basal':
            label = basal_segs[frame,Z,Y,X]
            if label > 0:
                tracked_nuc[frame,basal_segs[frame,...] == label] = spot['TrackID']
            else:
                label = suprabasal_segs[frame,Z,Y,X]
                if label > 0:
                    print(f'\n Nuc: {frame}, Basal is in Suprabasal: {spot.TrackID}')
                    tracked_nuc[frame,suprabasal_segs[frame,...] == label] = spot['TrackID']
                else:
                    print(f'\n Nuc: {frame}, {spot.TrackID}')
                    sli = get_cube_fill_as_slice(tracked_nuc[frame,...].shape,
                                                 np.array([Z,Y,X]))
                    tracked_nuc[frame,sli[0],sli[1],sli[2]] = spot['TrackID']
                    
            label = cyto_segs[frame,Z,Y,X]
            if label > 0:
                tracked_cyto[frame,cyto_segs[frame,...] == label] = spot['TrackID']
                
        elif spot['Cell type'] == 'Suprabasal':

            label = suprabasal_segs[frame,Z,Y,X]
            if label > 0:
                tracked_nuc[frame,suprabasal_segs[frame,...] == label] = spot['TrackID']
            else:
                label = basal_segs[frame,Z,Y,X]
                if label > 0:
                    print(f'\n Nuc: {frame}, Suprabasal is in Basal: {spot.TrackID}')
                    tracked_nuc[frame,basal_segs[frame,...] == label] = spot['TrackID']
                else:
                    print(f'\n Nuc: {frame}, {spot.TrackID}')
                    sli = get_cube_fill_as_slice(tracked_nuc[frame,...].shape,
                                                 np.array([Z,Y,X]))
                    tracked_nuc[frame,sli[0],sli[1],sli[2]] = spot['TrackID']
                
            # These will currently break the cyto
            # label = cyto_supra[frame,Z,Y,X]
            # if label > 0:
            #     tracked_cyto[frame,cyto_supra[frame,...] == label] = spot['TrackID']
            # else:
            #     print(f'\n Cyto: {frame}, {spot.TrackID}')
            #     sli = get_cube_fill_as_slice(tracked_cyto[frame,...].shape,
            #                                  np.array([Z,Y,X]))
            #     tracked_cyto[frame,sli[0],sli[1],sli[2]] = spot['TrackID']


# Save compressed (1Gb->10Mb)
tifffile.imwrite(path.join(dirname,'Mastodon/tracked_cyto.tif'),tracked_cyto
                 ,compression='zlib')

#%% Put back in all the basal nuc segs that don't have tracks

from imageUtils import most_likely_label

maxID = tracks[-1].TrackID.iloc[0]
for t in range(15):
    
    df = pd.DataFrame( measure.regionprops_table(basal_segs[t], properties=['label','centroid']) )
    df = df.rename(columns={'label':'NucID',
                            'centroid-0':'Z',
                            'centroid-1':'Y',
                            'centroid-2':'X'}).round().astype(int)
    df['TrackID'] = corresponding_trackIDs = np.array(
        [tracked_nuc[t,x[0],x[1],x[2]] for x in df[['Z','Y','X']].values])
    df = df[df['TrackID'] == 0]
    
    for _,row in df.iterrows():
        tracked_nuc[t, basal_segs[t] == row.NucID] = maxID
        maxID += 1
    
tifffile.imwrite(path.join(dirname,'Mastodon/tracked_nuc.tif'),tracked_nuc,
                 compression='zlib')





