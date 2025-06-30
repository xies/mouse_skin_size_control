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

all_df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.pkl'))
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


adjacent_tracks = [np.load(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),
                   allow_pickle=True).item() for t in range(15)]

#%% Find graph differences -- manually inspect neighborhood changes


# for t in range(14):
    
#     print(f'Frame: {t} -> {t+1}')
    
#     # Filter for border cells
#     g1 = adjacent_tracks[t]
#     prev_cells = np.array(list(g1.keys()))
#     prev_cells = set(prev_cells[~np.array([all_df.loc[t,tID]['Border','Meta'] for tID in prev_cells])])
    
#     g2 = adjacent_tracks[t+1]
#     this_cells = np.array(list(g2.keys()))
#     this_cells = set(this_cells[~np.array([all_df.loc[t+1,tID]['Border','Meta'] for tID in this_cells])])
    
#     cells_added = this_cells - prev_cells
#     cells_lost = prev_cells - this_cells
    
#     # Go through and make sure the 'added cells' are being born via MotherID
#     added_but_not_born = []
#     for cell in cells_added:
#         if np.isnan(all_df.loc[t+1,cell]['Mother','Meta']):
#             # Don't record if the 'adding' came from moving in from border
#             if (t,cell) in all_df.index and all_df.loc[t,cell]['Border','Meta']:
#                 continue
#             added_but_not_born.append(cell)
#     print(f'--- Added but not born: {added_but_not_born} ---')
    
#     lost_but_not_divided = []
#     for cell in cells_lost:
#         if np.isnan(all_df.loc[t,cell]['Daughter a','Meta']):
#             lost_but_not_divided.append(cell)
    
#     lost_not_divided_not_differentiated = []
#     for cell in lost_but_not_divided:
        
#         # If cell disappears completely
#         if (t+1,cell) not in all_df.index:
#             lost_not_divided_not_differentiated.append(cell)
#         elif all_df.loc[t+1,cell]['Cell type','Meta'] == 'Basal' and not all_df.loc[t+1,cell]['Border']:
#             lost_not_divided_not_differentiated.append(cell)
#     print(f'--- Lost but not divided or differentiated: {lost_not_divided_not_differentiated} ---')


#%%

#@todo: get # of differentiated cells in neighborhood

tracks = {trackID:t for trackID,t in all_df.reset_index().groupby('TrackID')}

for trackID,cf in tqdm(tracks.items()):
    
    # Look at -1 time frame, look at neighbors
    cf = cf.set_index('Frame',drop=True)
    cf['Num neighbor division 1 frame prior','Measurement'] = np.nan
    cf['Num neighbor delamination 1 frame prior','Measurement'] = np.nan

    for t in cf.index:
        if t > 0 and (cf.loc[t]['Cell type','Meta'] == 'Basal') and not cf.loc[t]['Border','Meta']:
            # if just born, look at mother cell
            if t > cf.iloc[0].name:
                trackID2look = int(trackID)
            elif not cf.iloc[0].Born.values[0]:
                print('---')
                continue
            else:
                trackID2look = int(cf.iloc[0]['Mother'].values[0])
                
            if trackID2look == 461 and t == 6: #@todo: this needs fixing...
                continue
            
            if (all_df.loc[t-1,trackID2look]['Cell type','Meta'] == 'Basal'):
                prev_neighbors = adjacent_tracks[t-1][trackID2look]
                prev_neighbors = all_df.loc[list(zip([t-1]*len(prev_neighbors),prev_neighbors))]
                num_prev_neighbor_divided = prev_neighbors['Divide next frame','Meta'].sum()
                num_prev_neighbor_diff = prev_neighbors['Delaminate next frame','Meta'].sum()
                cf.loc[t, ('Num neighbor division 1 frame prior','Measurement')] = \
                           num_prev_neighbor_divided
                cf.loc[t, ('Num neighbor delamination 1 frame prior','Measurement')] = \
                    num_prev_neighbor_diff
                # stop
                
    tracks[trackID] = cf.reset_index()

#%% Go forward in time -- put daughter info
 
for trackID,cf in tqdm(tracks.items()):
    
    cf = cf.set_index('Frame',drop=True)
    cf['Num daughter differentiated','Meta'] = np.nan
    daughterID_A = cf.iloc[0]['Daughter a','Meta']
    daughterID_B = cf.iloc[0]['Daughter b','Meta']
    if not np.isnan(daughterID_A):
        daughterA = tracks[daughterID_A]
        daughterB = tracks[daughterID_B]
        if daughterA.iloc[0]['Fate known','Meta'] and daughterB.iloc[0]['Fate known','Meta']:
            cf['Num daughter differentiated','Meta'] = int(daughterA.iloc[0]['Will differentiate','Meta']) \
                + int(daughterB.iloc[0]['Will differentiate','Meta'])
    tracks[trackID] = cf.reset_index()

tracks = pd.concat(tracks,ignore_index=True).set_index(['Frame','TrackID']).drop_duplicates()
Idaughter_fate_known = ~np.isnan(tracks['Num daughter differentiated','Meta'])
tracks.loc[Idaughter_fate_known,('At least one differentiated daughter','Meta')] = \
    tracks.loc[Idaughter_fate_known,('Num daughter differentiated','Meta')] > 0
tracks.loc[Idaughter_fate_known,('Both differentiated daughters','Meta')] = \
    tracks.loc[Idaughter_fate_known,('Num daughter differentiated','Meta')] == 2

#%% Save to pickle

tracks.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback_history.pkl'))
    
    