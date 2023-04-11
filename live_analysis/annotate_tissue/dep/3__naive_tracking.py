#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 23:46:10 2022

@author: xies
"""

import numpy as np
from skimage import io, measure
from glob import glob
from os import path
from scipy.spatial import distance
from re import findall
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm


dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

DEBUG = True
WEIGHT_OVERLAP = 0.8
WEIGHT_DISTANCE = 0.4


def sort_by_t(f):
    t = findall('t([0-9]+).tif',f)
    return int(t[0])


def compute_mask_overlaps(this_im,next_im,df_this):
    '''
    Returns the number of pixels in NEXT_IM that overlaps with each element of THIS_IM
    
    RETURNS:
        OVERLAPS: NxM matrix, N- number of objects in THIS_IM
        M - number of objects in NEXT_IM
    '''
    
    thisIDs = np.unique(this_im)[1:]
    nextIDs = np.unique(next_im)[1:]
    
    overlaps = pd.DataFrame(index=thisIDs, columns=nextIDs,dtype=float, data=0.)
    
    for label in thisIDs:
        # Basically use this label's mask to mask out the next_im, then histogram the label values that are nonzero
        mask = this_im == label
        total_area = df_this[df_this['label'] == label].iloc[0]['area']
        overlap_im = next_im[mask]
        
        overlappingIDs,pixel_counts = np.unique(overlap_im[overlap_im > 0], return_counts=True )
        
        for i,overlapID in enumerate(overlappingIDs):
            overlaps.loc[label,overlapID] = pixel_counts[i]/total_area
    
    return overlaps

def detect_divisions(df_this,df_next, size_ratio_threshold = -0.4):
    
    newborn = []
    for _, _this in df_this.iterrows():
        
        if np.isnan(_this['TrackID']):
            continue
        
        I = df_next['TrackID'] == _this['TrackID']
        _next = df_next[I]
        if len(_next) == 0:
            continue
        
        # Check if area is decreasing by more than given threshold
        if (_next['area'].values - _this['area']) / float(_this['area']) < size_ratio_threshold:
            newborn.append(_next.iloc[0]['TrackID'])
            
    return newborn

#%% Overlap track, detect division, manage label range

current_label = 1 #NB: set this at the beginning of time loop
T = 15

for t in tqdm(range(T-1)):

    if t == 0:
        this_frame = io.imread(path.join(dirname,'3d_nuc_seg/cellpose_cleaned_manual/t0.tif'))
    else:
        this_frame = io.imread(path.join(dirname,f'3d_nuc_seg/naive_tracking/t{t+1}.tif')) # load from /naive_tracking/ directory
    next_frame = io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t+1}.tif'))
    
    df_this = pd.DataFrame(measure.regionprops_table(this_frame,
                                                     properties = ['area','centroid','label',
                                                                   'euler_number']))
    df_next = pd.DataFrame(measure.regionprops_table(next_frame,
                                                     properties = ['area','centroid','label',
                                                                   'euler_number']))
    
    if t == 0:
        df_this['TrackID'] = np.nan
    else:
        df_this['TrackID'] = df_this['label']
    df_next['TrackID'] = np.nan
    
    # initialize tracked final images
    this_tracked = this_frame
    next_tracked = np.zeros(next_frame.shape,dtype=np.int16)
    
    overlaps = compute_mask_overlaps(this_frame,next_frame,df_this)
    
    for i,thisID in enumerate(df_this['label']):
        
        #@todo: currently reliant on overlap
        J = (-overlaps.loc[thisID].values).argmin()
        nextID = df_next.iloc[J]['label']
        mask = next_frame == nextID
        
        # Record onto image
        if t==0: # reindex the first image's lineageID for better range management / avoid using int32
            label2use = current_label
            current_label += 1
        else:
            label2use = thisID
        # if t == 4 and label2use == 20:
        #     error()
        if t == 0:
            this_tracked[this_frame == thisID] = label2use
            df_this.at[i,'TrackID'] = label2use
            
        next_tracked[mask] = label2use
        df_next.at[df_next['label'] == nextID,'TrackID'] = label2use
        
    # Now identify the next frame objects that were not included; put them back in and have a running tally of unused
    unaccounted_forIDs = np.unique(next_frame[ ~(next_tracked > 0) ])[1:]
    for nextID in unaccounted_forIDs:
        mask = next_frame == nextID
        next_tracked[mask] = current_label
        df_next.at[df_next['label'] == nextID,'TrackID'] = current_label
        current_label += 1
        # print(f'{nextID} became {current_label - 1}')
        
    # Detect division events and start new trackID
    df_this = pd.DataFrame(measure.regionprops_table(this_tracked, properties = ['area','label']))
    df_next = pd.DataFrame(measure.regionprops_table(next_tracked, properties = ['area','label']))
    
    df_this = df_this.rename(columns={'label':'TrackID'})
    df_next = df_next.rename(columns={'label':'TrackID'})
    
    newbornIDs = detect_divisions(df_this,df_next)
    
    
    for trackID in newbornIDs:
        mask = next_tracked == trackID
        next_tracked[mask] = current_label
        current_label += 1
        # print(f'newborn {trackID} became {current_label - 1}')
    
    if t == 0:
        io.imsave(path.join(dirname,f'3d_nuc_seg/naive_tracking/t{t}.tif'),this_tracked.astype(np.uint16),check_contrast=False)
    
    io.imsave(path.join(dirname,f'3d_nuc_seg/naive_tracking/t{t+1}.tif'),next_tracked.astype(np.uint16),check_contrast=False)
    
    
    