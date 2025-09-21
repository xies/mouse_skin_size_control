#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 13:13:12 2025

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io
from measurements import extract_nuc_and_cell_and_microenvironment_mask_from_idx
from imageUtils import trim_multimasks_to_shared_bounding_box, pad_image_to_size_centered, create_average_object_from_multiple_masks

model_dir = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Lineage models/'


# Load all datasets
dirnames = {'R1':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/',
           'R2':'/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'}
all_df = []
for name,dirname in dirnames.items():
    _df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback_history.pkl'))
    _df = _df.drop_duplicates().sort_index().reset_index()
    _df['TrackID'] = name + '_' + _df['TrackID'].astype(str)
    _df = _df.set_index(['Frame','TrackID'])
    _df['Region'] = name
    all_df.append(_df)

all_df = pd.concat(all_df)
all_tracks = {trackID:t for trackID,t in all_df.reset_index().groupby('TrackID')}

# Load PCA datasets
name = 'divisions'
components = pd.read_pickle( path.join(model_dir,f'Probabilistic PCA/{name}/components.pkl'))
transformed = pd.read_pickle( path.join(model_dir,f'Probabilistic PCA/{name}/transformed.pkl'))
pca = transformed.xs('PCA',level=1,axis=1)
pca.columns = range(pca.shape[1])

#%%

# Load images
tracked_cyto_by_region = {name: io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif')) for name,dirname in dirnames.items()}
tracked_nuc_by_region = {name: io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif')) for name,dirname in dirnames.items()}
adjdict_by_region = {name: [np.load(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),allow_pickle=True).item() for t in range(15)] for name,dirname in dirnames.items()}

#%%

standard_size = (30,150,150)

def get_display_of_average_nuc_cyto_micro(ncm_list,largest_size=None):
    nuc = [m[0] for m in ncm_list]
    cyto = [m[1] for m in ncm_list]
    micro = [m[2] for m in ncm_list]
    nuc = create_average_object_from_multiple_masks(nuc,prealign=False)
    cyto = create_average_object_from_multiple_masks(cyto,prealign=False)
    micro = create_average_object_from_multiple_masks(micro,prealign=False)

    if largest_size is None:
        largest_size = np.array((nuc.shape,cyto.shape,micro.shape)).max(axis=0)
    nuc = pad_image_to_size_centered(nuc,largest_size)
    cyto = pad_image_to_size_centered(cyto,largest_size)
    micro = pad_image_to_size_centered(micro,largest_size)
    return np.stack((nuc,cyto,micro))

#%%

PC = 3
n_exmaples = 20

# Pull out 3 examplar 'PC0' cells
pc0_exemplars = pca[PC].sort_values().iloc[-n_exmaples:].index[::-1]
pc0_antiexemplars = pca[PC].sort_values().iloc[:n_exmaples].index

pc0_masks = [extract_nuc_and_cell_and_microenvironment_mask_from_idx(
                      idx,adjdict_by_region,tracked_nuc_by_region,tracked_cyto_by_region)
                      for idx in pc0_exemplars]
pc0_antimasks = [extract_nuc_and_cell_and_microenvironment_mask_from_idx(
                      idx,adjdict_by_region,tracked_nuc_by_region,tracked_cyto_by_region)
                      for idx in pc0_antiexemplars]

pc0_mean = get_display_of_average_nuc_cyto_micro(pc0_masks,standard_size)
io.imsave(path.join(model_dir,f'Probabilistic PCA/{name}/pc{PC}_example.tif'),pc0_mean)
pc0_antimean = get_display_of_average_nuc_cyto_micro(pc0_antimasks,standard_size)
io.imsave(path.join(model_dir,f'Probabilistic PCA/{name}/pc{PC}_antiexample.tif'),pc0_antimean)
