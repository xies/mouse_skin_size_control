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

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.csv'),index_col=[0,1])
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

#%% Find graph differences -- manually inspect neighborhood changes

adjacent_tracks = [np.load(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),
                   allow_pickle=True).item() for t in range(15)]

for t in range(14):
    
    print(f'Frame: {t} -> {t+1}')
    
    # Filter for border cells
    g1 = adjacent_tracks[t]
    prev_cells = np.array(list(g1.keys()))
    prev_cells = set(prev_cells[~np.array([all_df.loc[t,tID]['Border'] for tID in prev_cells])])
    
    g2 = adjacent_tracks[t+1]
    this_cells = np.array(list(g2.keys()))
    this_cells = set(this_cells[~np.array([all_df.loc[t+1,tID]['Border'] for tID in this_cells])])
    
    cells_added = this_cells - prev_cells
    cells_lost = prev_cells - this_cells
    
    # Go through and make sure the 'added cells' are being born via MotherID
    added_but_not_born = []
    for cell in cells_added:
        if np.isnan(all_df.loc[t+1,cell]['Mother']):
            # Don't record if the 'adding' came from moving in from border
            if (t,cell) in all_df.index and all_df.loc[t,cell]['Border']:
                continue
            added_but_not_born.append(cell)
    print(f'--- Added but not born: {added_but_not_born} ---')
    
    lost_but_not_divided = []
    for cell in cells_lost:
        if np.isnan(all_df.loc[t,cell]['Daughter a']):
            lost_but_not_divided.append(cell)
    
    lost_not_divided_not_differentiated = []
    for cell in lost_but_not_divided:
        
        # If cell disappears completely
        if (t+1,cell) not in all_df.index:
            lost_not_divided_not_differentiated.append(cell)
        elif all_df.loc[t+1,cell]['Cell type'] == 'Basal' and not all_df.loc[t+1,cell]['Border']:
            lost_not_divided_not_differentiated.append(cell)
    print(f'--- Lost but not divided or differentiated: {lost_not_divided_not_differentiated} ---')

#%%



#%% Look for connected components differences

# import networkx as nx

# t = 0

# added = dict()
# lost = dict()
# for edge in adjacent_tracks[t].keys():
    
#     if all_df.loc[t,edge]['Border'] or all_df.loc[(t,edge),'Divide next frame']:
#         continue
#     if all_df.loc[t+1,edge]['Cell type'] == 'Suprabasal':
#         continue
    
#     connected_at_t = set(adjacent_tracks[t][edge])
#     connected_at_tplus1 = set(adjacent_tracks[t+1][edge])
    
#     added[edge] = connected_at_tplus1 - connected_at_t
#     lost[edge] = connected_at_t - connected_at_tplus1
    
#     added
    



#%%

#@todo: get # of differentiated cells in neighborhood

subset_fields = ['Nuclear volume','Cell volume']

tracks = []
for _,cf in all_df.reset_index().groupby('TrackID'):

    offset_cf = get_time_offset_data(cf,subset_fields,offset_by=-1)
    if offset_cf is not None:
        cf = cf.reset_index(drop=True).set_index('Frame')
        cf = cf.join(offset_cf)
    tracks.append(cf)

tracks = pd.concat(tracks,ignore_index=True).set_index(['Frame','TrackID'])


    
    
    
    