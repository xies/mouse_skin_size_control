#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:49:18 2025

@author: xies
"""
 
# Core libraries
import numpy as np
# from skimage import io, measure, morphology
import pandas as pd
import matplotlib.pylab as plt

# Specific utils
from imageUtils import draw_labels_on_image, draw_adjmat_on_image, \
    most_likely_label, colorize_segmentation
from mathUtils import get_neighbor_idx, argsort_counter_clockwise

# General utils
from tqdm import tqdm
from os import path

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_aggregated.csv'),index_col=[0,1])
all_trackIDs = all_df.reset_index()

def get_time_offset_data(cf:pd.DataFrame,subset_fields,offset_by=-1):
    
    if len(cf) <= abs(offset_by):
        X = np.ones( (len(cf), len(subset_fields)) )*np.nan
        offset_data = pd.DataFrame(X,
                                   columns=[f'{field} offset {offset_by}' for field in subset_fields],
                                   index=cf.index)
    else:
        
        cf = cf.sort_values(by='Frame',ascending=True).set_index('Frame')
        idx = cf.index -1
        idx = idx[ np.isin(idx, cf.index)]
        X = cf.loc[idx, subset_fields].values
        if offset_by < 0:
            X = np.vstack(( np.ones( (abs(offset_by), len(subset_fields)) )*np.nan, X ))
        else:
            X = np.vstack(( X, np.ones( (abs(offset_by), len(subset_fields)) )*np.nan ))
        print(X.shape)
        print(cf.index)
        offset_data = pd.DataFrame(X,
                                   columns=[f'{field} offset {offset_by}' for field in subset_fields],
                                   index=cf.index)
        
    return offset_data

#%% Find graph differences

adjacent_tracks = [np.load(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),
                   allow_pickle=True).item() for t in range(15)]

# Filter for border cells

g1 = adjacent_tracks[0]
g2 = adjacent_tracks[1]
border1 = all_df.xs(0,level='Frame').index

prev_cells = set(g1.keys())
this_cells = set(g2.keys())

cells_added = this_cells - prev_cells
cells_lost = prev_cells - this_cells

#%%


subset_fields = ['Nuclear volume','Cell volume']

tracks = []
for _,cf in all_df.reset_index().groupby('TrackID'):

    offset_cf = get_time_offset_data(cf,subset_fields,offset_by=-1)
    if offset_cf is not None:
        cf = cf.reset_index(drop=True).set_index('Frame')
        cf = cf.join(offset_cf)
    tracks.append(cf)


tracks = pd.concat(tracks,ignore_index=True).set_index(['Frame','TrackID'])


    
    
    
    