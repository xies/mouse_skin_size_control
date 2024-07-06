#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 22:06:01 2023

@author: xies
"""

from skimage import io
from os import path
from tqdm import tqdm

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1'

#%% Filter the tracking predictions (+daughters) by ground truth

cyto_track = io.imread(path.join(dirname,'manual_basal_tracking/basal_tracks_cyto.tif'))
daughter_track = io.imread(path.join(dirname,'manual_basal_tracking_daughters/basal_tracking_daughters_cyto.tif'))

seg_notrack = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))

filtered = np.zeros_like(seg_notrack)
filtered_daughters = np.zeros_like(seg_notrack)
for t in tqdm(range(15)):
    
    # Cells
    basalIDs = np.unique(cyto_track[t,...])[1:]
    for bID in basalIDs:
        this_nuc_labels = seg_notrack[t,...]
        masked = this_nuc_labels[cyto_track[t,...] == bID]
        unique,counts = np.unique(masked[masked>0],return_counts=True)
        most_likely_nuc_label = unique[counts.argmax()]
        
        mask = seg_notrack[t,...] == most_likely_nuc_label
        filtered[t,mask] = bID
        
    # Daughters
    daughterIDs = np.unique(daughter_track[t,...])[1:]
    for dID in daughterIDs:
        this_nuc_labels = seg_notrack[t,...]
        masked = this_nuc_labels[daughter_track[t,...] == dID]
        unique,counts = np.unique(masked[masked>0],return_counts=True)
        most_likely_nuc_label = unique[counts.argmax()]
        
        mask = seg_notrack[t,...] == most_likely_nuc_label
        filtered[t,mask] = dID
        
io.imsave(path.join(dirname,'manual_basal_tracking/basal_track_nuclei.tif'),filtered.astype(np.uint16))
io.imsave(path.join(dirname,'manual_basal_tracking_daughters/basal_track_daughter_nuclei.tif'),filtered_daughters.astype(np.uint16))

#%% Visual insepction

test_tracking = io.imread(path.join(dirname,'ultracking/ultracking.tif'))
ground_truth = io.imread(path.join(dirname,'ultracking/manual_basal_nuclei_track.tif'))

filtered = np.zeros_like(test_tracking)

mask = ground_truth > 0

all_candidates = np.unique(test_tracking[mask])[1:]
for ID in tqdm(all_candidates):
    filtered = filtered + (test_tracking == ID)*ID

io.imsave(path.join(dirname,'ultracking/filtered_ultracking.tif'),filtered.astype(np.uint16))

