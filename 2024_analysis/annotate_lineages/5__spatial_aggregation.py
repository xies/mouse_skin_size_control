#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  4 15:43:28 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io, measure, morphology
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import tifffile

# Specific utils
from imageUtils import most_likely_label

# General utils
from tqdm import tqdm
from os import path

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

def find_touching_labels(labels, centerID, threshold, selem=morphology.disk(3)):
    this_mask = labels == centerID
    this_mask_dil = morphology.binary_dilation(this_mask,selem)
    touchingIDs,counts = np.unique(labels[this_mask_dil],return_counts=True)
    touchingIDs[counts > threshold] # should get rid of 'conrner touching'

    touchingIDs = touchingIDs[touchingIDs > 2] # Could touch background pxs
    touchingIDs = touchingIDs[touchingIDs != centerID] # nonself
    
    return touchingIDs
    

#% Reconstruct adj network from cytolabels that touch
def get_adjdict_from_2d_segmentation(seg2d:np.array, touching_threshold:int = 2):
    '''

    Parameters
    ----------
    seg2d : np.array
        2D cytoplasmic segmentation on which to determine adjacency
    touching_threshold : int, optional
        Minimum number of overlap pixels. The default is 2.

    Returns
    -------
    A : dict
        Dictionary of adjacent labels:
            {centerID : neighborIDs }

    '''
    #@todo: OK for 3D segmentation? currently no...
    assert(seg2d.ndim == 2) # only works with 2D images for now
    
    A = {centerID:find_touching_labels(seg2d, centerID, touching_threshold)
         for centerID in np.unique(seg2d)[1:]}
    
    return A

#%% Build basal adj graph
# @todo: suprabasal adj could also be annotated but probably use delauney3D instead?

# Load flat 2D adj image (omits suprabasal cells)
seg2d = [io.imread(path.join(dirname,f'Image flattening/flat_cyto_seg_manual/t{t}.tif'))
         for t in range(15)]

# Load basal 3D nuclei
nuc = [io.imread(path.join(dirname,f'3d_nuc_seg/cellpose_cleaned_manual/t{t}_basal.tif'))
       for t in range(15)]

# Load all basal + suprabasal tracked cells
tracked_nuc = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
adjDicts = [get_adjdict_from_2d_segmentation(seg) for seg in seg2d]

#%% Connect the (frame, adjID) into TrackID
# Map adjID -> NucID (dense basal nuc 3d) -> TrackID (all tracked from t=0)

label_transfers = [] # adjID -> TrackID
for t in range(15):
    
    this_transfer = pd.DataFrame(measure.regionprops_table(nuc[t].max(axis=0), intensity_image=seg2d[t],
                              properties = ['label'],extra_properties=[most_likely_label] ))
    this_transfer = this_transfer.rename(columns={'label':'NucID',
                                                      'most_likely_label':'adjID'})
    
    _this_transfer = pd.DataFrame(measure.regionprops_table(nuc[t], intensity_image=tracked_nuc[t,...],
                              properties = ['label'],extra_properties=[most_likely_label] ))
    _this_transfer = _this_transfer.rename(columns={'label':'NucID',
                                                      'most_likely_label':'TrackID'})
    this_transfer = this_transfer.merge(_this_transfer)
    this_transfer['Frame'] = t
    label_transfers.append(this_transfer)
    
label_transfers = pd.concat(label_transfers)
label_transfers = label_transfers.set_index(['Frame','NucID'])

# Input manually NucID is indexed
label_transfers.loc[(0,217),'adjID'] = 334
label_transfers.loc[(1,361),'adjID'] = 335
label_transfers.loc[(1,441),'adjID'] = 263
label_transfers.loc[(1,448),'adjID'] = 358
label_transfers.loc[(2,109),'adjID'] = 358
label_transfers.loc[(2,318),'adjID'] = 243
label_transfers.loc[(4,776),'adjID'] = 132
label_transfers.loc[(4,916),'adjID'] = 139
label_transfers.loc[(4,1106),'adjID'] = 8
label_transfers.loc[(5,834),'adjID'] = 262
label_transfers.loc[(5,3285),'adjID'] = 107
label_transfers.loc[(5,845),'adjID'] = 367
label_transfers.loc[(6,769),'adjID'] = 367
label_transfers.loc[(6,1131),'adjID'] = 225
label_transfers.loc[(7,602),'adjID'] = 376
label_transfers.loc[(7,603),'adjID'] = 358
label_transfers.loc[(7,604),'adjID'] = 13
label_transfers.loc[(7,735),'adjID'] = 271
label_transfers.loc[(8,666),'adjID'] = 319
label_transfers.loc[(8,814),'adjID'] = 316
label_transfers.loc[(10,651),'adjID'] = 286
label_transfers.loc[(11,614),'adjID'] = 364
label_transfers.loc[(11,636),'adjID'] = 378
label_transfers.loc[(12,742),'adjID'] = 355
label_transfers.loc[(14,835),'adjID'] = 197
label_transfers.loc[(12,992),'adjID'] = 377
label_transfers.loc[(12,791),'adjID'] = 356
label_transfers.loc[(12,791),'adjID'] = 356

# Detect NucID that has no AdjID (unmapped)
print( ' --- Missing from basal adj graph ---')
print( label_transfers.loc[np.isnan(label_transfers['adjID'])] )

# Detect duplicated AdjID (degeneracy)
print( ' --- Duplicated in basal adj graph ---')
label_transfers = label_transfers.reset_index()
print( label_transfers[label_transfers.duplicated(subset=['Frame','adjID'])] )
label_transfers = label_transfers.set_index(['Frame','NucID'])

# Detect missing AdjID 
missingIDs = []
print( ' --- Missing from label transfers ---')
for t in range(15):
    missingIDs.append( list(set(list(adjDicts[t].keys())) - 
                            set(label_transfers.xs(t,level='Frame')['adjID'].values)) )
print( missingIDs )

label_transfers = label_transfers.reset_index().set_index(['Frame','adjID']).astype(int)

# Go through the AdjDict and swap adjID for trackID
# Drop the missing adjIDs
adjacent_tracks = []
for t in tqdm(range(15)):
    this_frame_adj = {}
    for k,val in adjDicts[t].items():
        if not k in missingIDs[t]:
            new_key = label_transfers.loc[(t,k), 'TrackID'].item()
            new_values = np.array(
                [label_transfers.loc[(t,v),'TrackID'].item() for v in val if v not in missingIDs[t]])
            this_frame_adj[new_key] = new_values
    adjacent_tracks.append(this_frame_adj)
    np.save(path.join(dirname,f'Mastodon/basal_connectivity_3d/adjacenct_trackIDs_t{t}.npy'),this_frame_adj)

#%% Draw trackID

from imageUtils import draw_adjmat_on_image_3d, adjdict_to_mat

for t in range(15):
    
    coords_3d = pd.DataFrame(measure.regionprops_table(tracked_nuc[t,...], properties=['label','centroid']))
    coords_3d = coords_3d.rename(columns={'label':'TrackID','centroid-0':'Z','centroid-1':'Y','centroid-2':'X',}).set_index('TrackID')
    A = adjdict_to_mat(adjacent_tracks[t])
    im = draw_adjmat_on_image_3d(A, coords_3d, im_shape = nuc[0].shape, line_width=2, colorize=True)
    tifffile.imwrite(path.join(dirname,f'Mastodon/basal_connectivity_3d/t{t}_conn_3d.tif'),im,
                     compression='zlib')
    
im_conn = np.array([io.imread(path.join(dirname,f'Mastodon/basal_connectivity_3d/t{t}_conn_3d.tif')) for t in range(15)])
tifffile.imwrite(path.join(dirname,'Mastodon/basal_connectivity_3d/basal_connectivity_3d.tif'),
                 im_conn.astype(np.uint16),compression='zlib')

#%% Aggregate over adjacency network

from collections.abc import Callable
def aggregate_over_adj(adj: dict, aggregators: dict[str,Callable],
                       df = pd.DataFrame, fields2aggregate=list[str]):
    df_aggregated = pd.DataFrame()

    for agg_name in aggregators.keys():
        for field in fields2aggregate:
            df_aggregated[f'{agg_name} adjac {field}'] = np.nan
            
    for centerID,neighborIDs in adj.items():
        neighbors = df.loc[neighborIDs]
        if len(neighbors) > 0:
            for agg_name, agg_func in aggregators.items():
                for field in fields2aggregate:
                    df_aggregated.loc[ (centerID,f'{agg_name} adjac {field}') ] = \
                        agg_func(neighbors[field].values)
                        
    df_aggregated.index.name = 'TrackID'
    
    return df_aggregated.reset_index()

aggregators = {'Mean':np.nanmean,
               'Median':np.nanmedian,
               'Max':np.nanmax,
               'Min':np.nanmin,
               'Std':np.nanstd}

#@todo: load list of fields
fields2aggregate = ['Nuclear volume','Height to BM','Cell volume','Basal area','Apical area']
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics.csv'),index_col=['Frame','TrackID'])

aggregated_fields = []
for t in tqdm(range(15)):
    
    adj = adjacent_tracks[t]
    df_agg = aggregate_over_adj(adj, aggregators, all_df.xs(t,level='Frame'), fields2aggregate)
    df_agg['Num basal neighbors'] = df_agg['TrackID'].map({k:len(v) for k,v in adj.items()})
    df_agg['Frame'] = t
    aggregated_fields.append(df_agg)
    
    
aggregated_fields = pd.concat(aggregated_fields,ignore_index=True)
aggregated_fields = aggregated_fields.set_index(['Frame','TrackID'])
all_df = all_df.join(aggregated_fields)

all_df.to_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.csv'))

#%% Lookbacks

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.csv'),
                     index_col=['Frame','TrackID']).sort_index()

from functools import reduce

tracks = [t for _,t in all_df.groupby('TrackID')]
def lookback(tracks: list[pd.DataFrame], fields2lookback:list[str], num_frames_lookback:str=1):
    df_lookback = []

    for track in tracks:
        _track = pd.DataFrame(index=track.index,
                              columns = [f'{field} at {num_frames_lookback} frame prior' 
                                         for field in fields2lookback])
        for field in fields2lookback:
            v = track[field].values
            v = np.insert(v,0,np.nan)
            v = v[:len(v)-1]
            
            _track[f'{field} at {num_frames_lookback} frame prior'] = v
        
        df_lookback.append(_track)
        
    return pd.concat(df_lookback)

# Grab all fields that has match
fields2lookback = set(reduce(list.__add__, [all_df.columns[all_df.columns.str.contains(query)].tolist()
    for query in ['Nuclear volume','Height to BM','adjac']] ))

df_lookback = lookback(tracks,fields2lookback)
all_df = all_df.join(df_lookback)

all_df.to_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.csv'))







