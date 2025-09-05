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
from basicUtils import sort_by_timestamp

# General utils
from tqdm import tqdm
from os import path
import pickle as pkl

dx = 0.25
dz = 1

# Filenames??
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)


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

#%% Flatten segmentation from the heightmap

print('Flattening cytoplasmic segmentation map into ')

TOP_OFFSET = -30 #NB: top -> more apical but lower z-index
BOTTOM_OFFSET = 10

imstack = io.imread(path.join(dirname,'Mastodon/tracked_cyto.tif'))
T = imstack.shape[0]

for t in tqdm(range(T)):
    
    im = imstack[t,...]
    Z,XX,_ = im.shape
    
    heightmap = io.imread(path.join(dirname,f'Image flattening/heightmaps/t{t}.tif'))
    
    output_dir = path.join(dirname,'Image flattening/flat_tracked_cyto')
    
    flat = np.zeros((-TOP_OFFSET+BOTTOM_OFFSET,XX,XX))
    Iz_top = heightmap + TOP_OFFSET
    Iz_bottom = heightmap + BOTTOM_OFFSET
    
    for x in range(XX):
        for y in range(XX):
            
            flat_indices = np.arange(0,-TOP_OFFSET+BOTTOM_OFFSET)
            
            z_coords = np.arange(Iz_top[y,x],Iz_bottom[y,x])
            # sanitize for out-of-bounds
            z_coords[z_coords < 0] = 0
            z_coords[z_coords >= Z] = Z-1
            I = (z_coords > 0) & (z_coords < Z)
            
            flat[flat_indices[I],y,x] = im[z_coords[I],y,x]
    
    io.imsave( path.join(output_dir,f't{t}.tif'), flat.astype(np.uint16),check_contrast=False)

#%% Build basal adj graph from flattened segmentation
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
    
    this_transfer = pd.DataFrame(measure.regionprops_table(nuc[t].max(axis=0),
                                                           intensity_image=seg2d[t],
                              properties = ['label'],extra_properties=[most_likely_label] ))
    this_transfer = this_transfer.rename(columns={'label':'NucID',
                                                      'most_likely_label':'adjID'})
    
    _this_transfer = pd.DataFrame(measure.regionprops_table(nuc[t],
                                                            intensity_image=tracked_nuc[t,...],
                              properties = ['label'],extra_properties=[most_likely_label] ))
    _this_transfer = _this_transfer.rename(columns={'label':'NucID',
                                                      'most_likely_label':'TrackID'})
    this_transfer = this_transfer.merge(_this_transfer)
    this_transfer['Frame'] = t
    label_transfers.append(this_transfer)
    
label_transfers = pd.concat(label_transfers)
label_transfers = label_transfers.set_index(['Frame','NucID'])
label_transfers = label_transfers[(label_transfers['TrackID'] <= len(tracks) + 1)
                                  & (label_transfers['TrackID'] >0)]

# Input manually NucID is indexed
if dirname == '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/':
    label_transfers.loc[(1,361),'adjID'] = 335
    label_transfers.loc[(2,2119),'adjID'] = 354
    label_transfers.loc[(2,2119),'adjID'] = 394
    label_transfers.loc[(4,776),'adjID'] = 132
    label_transfers.loc[(4,916),'adjID'] = 139
    label_transfers.loc[(4,1106),'adjID'] = 8
    label_transfers.loc[(5,834),'adjID'] = 262
    label_transfers.loc[(5,845),'adjID'] = 367
    label_transfers.loc[(5,3285),'adjID'] = 107
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
    label_transfers.loc[(12,742),'adjID'] = 355
    label_transfers.loc[(12,791),'adjID'] = 356
    label_transfers.loc[(12,992),'adjID'] = 377
    label_transfers.loc[(14,927),'adjID'] = 373
    label_transfers.loc[(14,835),'adjID'] = 197
    
elif dirname == '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/':
    #W-R2
    label_transfers.loc[(0,897),'adjID'] = 197
    label_transfers.loc[(0,1175),'adjID'] = 368
    label_transfers.loc[(0,1004),'adjID'] = 164
    label_transfers.loc[(0,1162),'adjID'] = 364
    label_transfers.loc[(0,1035),'adjID'] = 341
    label_transfers.loc[(0,1205),'adjID'] = 362
    
    label_transfers.loc[(1,903),'adjID'] = 15
    label_transfers.loc[(1,1038),'adjID'] = 351
    label_transfers.loc[(1,1033),'adjID'] = 232
    label_transfers.loc[(1,991),'adjID'] = 238
    
    label_transfers.loc[(2,707),'adjID'] = 257
    label_transfers.loc[(2,1119),'adjID'] = 380
    label_transfers.loc[(2,1117),'adjID'] = 378
    label_transfers.loc[(2,807),'adjID'] = 323
    label_transfers.loc[(2,810),'adjID'] = 346
    label_transfers.loc[(2,846),'adjID'] = 358
    label_transfers.loc[(2,890),'adjID'] = 11
    label_transfers.loc[(2,984),'adjID'] = 363
    
    label_transfers.loc[(3,1046),'adjID'] = 265
    label_transfers.loc[(3,997),'adjID'] = 82
    
    label_transfers.loc[(4,633),'adjID'] = 313
    label_transfers.loc[(4,731),'adjID'] = 94
    label_transfers.loc[(4,751),'adjID'] = 388
    label_transfers.loc[(4,771),'adjID'] = 358
    label_transfers.loc[(4,819),'adjID'] = 367
    label_transfers.loc[(4,823),'adjID'] = 16
    label_transfers.loc[(4,979),'adjID'] = 36
    
    label_transfers.loc[(5,550),'adjID'] = 284
    label_transfers.loc[(5,606),'adjID'] = 272
    label_transfers.loc[(5,704),'adjID'] = 334
    label_transfers.loc[(5,877),'adjID'] = 6
    label_transfers.loc[(5,881),'adjID'] = 30
    
    label_transfers.loc[(6,597),'adjID'] = 271
    label_transfers.loc[(6,952),'adjID'] = 336
    label_transfers.loc[(6,627),'adjID'] = 329
    label_transfers.loc[(6,932),'adjID'] = 54
    label_transfers.loc[(6,993),'adjID'] = 974
    label_transfers.loc[(6,992),'adjID'] = 975
    label_transfers.loc[(6,995),'adjID'] = 976
    label_transfers.loc[(6,963),'adjID'] = 102
    label_transfers.loc[(6,710),'adjID'] = 326
    label_transfers.loc[(6,844),'adjID'] = 328
    label_transfers.loc[(6,951),'adjID'] = 288
    
    label_transfers.loc[(9,900),'adjID'] = 326
    label_transfers.loc[(9,902),'adjID'] = 327
    label_transfers.loc[(11,725),'adjID'] = 393
    label_transfers.loc[(11,940),'adjID'] = 346
    label_transfers.loc[(11,1003),'adjID'] = 179
    label_transfers.loc[(11,1068),'adjID'] = 125
    label_transfers.loc[(11,1154),'adjID'] = 404
    label_transfers.loc[(11,1119),'adjID'] = 333
    label_transfers.loc[(11,725),'adjID'] = 261
    label_transfers.loc[(11,1152),'adjID'] = 407
    label_transfers.loc[(11,1153),'adjID'] = 408
    
    label_transfers.loc[(12,906),'adjID'] = 277
    label_transfers.loc[(12,1010),'adjID'] = 279
    label_transfers.loc[(12,1082),'adjID'] = 146
    label_transfers.loc[(12,1122),'adjID'] = 290
    label_transfers.loc[(12,1166),'adjID'] = 381
    label_transfers.loc[(12,1179),'adjID'] = 385
    
    label_transfers.loc[(13,742),'adjID'] = 239
    label_transfers.loc[(13,775),'adjID'] = 283
    label_transfers.loc[(13,1057),'adjID'] = 381
    label_transfers.loc[(13,872),'adjID'] = 305
    label_transfers.loc[(13,1047),'adjID'] = 381
    label_transfers.loc[(13,1060),'adjID'] = 383
    label_transfers.loc[(13,1065),'adjID'] = 290
    label_transfers.loc[(13,1057),'adjID'] = 384
    
    label_transfers.loc[(14,967),'adjID'] = 368
    label_transfers.loc[(14,816),'adjID'] = 355
    label_transfers.loc[(14,962),'adjID'] = 272
    label_transfers.loc[(14,964),'adjID'] = 124
    label_transfers.loc[(14,964),'adjID'] = 372
    label_transfers.loc[(14,944),'adjID'] = 309
    label_transfers.loc[(14,935),'adjID'] = 373
else:
    error

label_transfers = label_transfers.dropna(subset='TrackID')
# Detect NucID that has no AdjID (unmapped)
# Go through these, some are borders but some are actually missing
print( ' --- Missing from basal adj graph ---')
print( label_transfers.loc[np.isnan(label_transfers['adjID'])] )

# Detect duplicated AdjID (degeneracy)
print( ' --- Duplicated in basal adj graph ---')
label_transfers = label_transfers.reset_index()
dups = label_transfers[label_transfers.duplicated(subset=['Frame','adjID'])].set_index('Frame')
print( label_transfers[label_transfers.duplicated(subset=['Frame','adjID'])] )
label_transfers = label_transfers.set_index(['Frame','NucID'])

# Detect missing AdjID  -> These are usually border cells and we can tolerate these
missingIDs = []
print( ' --- Missing: AdjIDs without NucID ---')
for t in range(15):
    missingIDs.append( list(set(list(adjDicts[t].keys())) - 
                            set(label_transfers.xs(t,level='Frame')['adjID'].values)) )
print( missingIDs )

#%%

label_transfers = label_transfers.reset_index().set_index(['Frame','adjID'])
label_transfers['TrackID'] = label_transfers['TrackID'].astype(int)

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

for t in tqdm(range(15)):
    
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
    
    df_aggregated = pd.DataFrame(
        columns = [f'{k} adjac {f}' for k in aggregators.keys() for f in fields2aggregate],
        index=df.index, dtype=float)

    # for agg_name in aggregators.keys():
    #     for field in fields2aggregate:
    #         df_aggregated[f'{agg_name} adjac {field}'] = np.nan
        
    for centerID,neighborIDs in adj.items():
        neighbors = df.loc[neighborIDs]
        if len(neighbors) > 0:
            for agg_name, agg_func in aggregators.items():
                for field in fields2aggregate:
                    if neighbors[field].values.dtype == float:
                        if not np.all(np.isnan(neighbors[field].values)):
                            df_aggregated.loc[centerID,f'{agg_name} adjac {field}'] = \
                                agg_func(neighbors[field].values)
                    else:
                        df_aggregated.loc[centerID,f'{agg_name} adjac {field}'] = \
                            agg_func(neighbors[field].values)
    
    df_aggregated.index.name = 'TrackID'
    
    return df_aggregated.reset_index()

def frac_sphase(v):
    has_cell_cycle = v[v != 'NA']
    if len(has_cell_cycle) > 0:
        frac = (has_cell_cycle == 'SG2').sum() / len(has_cell_cycle)
    else:
        frac = np.nan
    return frac

all_df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics.pkl'))
tracks = {trackID:t for trackID,t in all_df.groupby('TrackID')}

aggregators = {'Mean':np.nanmean,
               'Median':np.nanmedian,
               'Max':np.nanmax,
               'Min':np.nanmin,
               'Std':np.nanstd}

# Aggregate every non-metadata field
fields2aggregate = all_df.xs('Measurement',axis=1,level=1).columns
# Drop Age, XYZ
fields2aggregate = fields2aggregate.drop(['X','Y','Z','Z-cyto','Y-cyto','X-cyto','Time',
                                          'Age','X-pixels','Y-pixels'] +
                                         [f for f in fields2aggregate if 'smoothed' in f])

aggregated_fields = []
for t in tqdm(range(15)):
    
    adj = adjacent_tracks[t]
    this_frame = all_df.xs(t,level='Frame')
    df_agg = aggregate_over_adj(adj, aggregators, this_frame, fields2aggregate)
    df_agg['Num basal neighbors'] = df_agg['TrackID'].map({k:len(v) for k,v in adj.items()})
    # @todo: one-off dist to neighbors df_agg['Mean distance to basal neighbors']
    # @todo: relative-to-mean
    df_relative = pd.DataFrame(index=df_agg.index,
                               columns=[f'Relative {field}' for field in fields2aggregate])
    for field in fields2aggregate:
        df_relative[f'Relative {field}'] = this_frame[field,'Measurement'].values / df_agg[f'Mean adjac {field}'].values
    
    df_agg['Frac of neighbors in S phase'] = aggregate_over_adj(adj,
                                                                {'Frac':frac_sphase},
                                                                all_df.xs(t,level='Frame'),
                                                                ['Cell cycle phase']).drop(columns='TrackID')
    
    df_agg['Frame'] = t
    df_agg = pd.concat((df_agg,df_relative),axis=1)
    aggregated_fields.append(df_agg)
    
aggregated_fields = pd.concat(aggregated_fields,ignore_index=True)
aggregated_fields = aggregated_fields.set_index(['Frame','TrackID'])
new_cols = pd.DataFrame()
new_cols['Name'] = aggregated_fields.columns
new_cols['Metadata'] = 'Measurement'
aggregated_fields.columns = pd.MultiIndex.from_frame(new_cols)

all_df = all_df.join(aggregated_fields)

all_df.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.pkl'))

 #%% Lookbacks

all_df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated.pkl')).sort_index()
tracks = {trackID:t for trackID,t in all_df.groupby('TrackID')}
all_df_with_supra = all_df.copy()

from functools import reduce

def lookback(tracks: list[pd.DataFrame], fields2lookback:list[str], num_frames_lookback:str=1):
    df_lookback = []

    for _,track in tqdm(tracks.items()):
        _track = pd.DataFrame(index=track.index,
                              columns = [f'{field} at {num_frames_lookback} frame prior' 
                                         for field in fields2lookback])
        for field in fields2lookback:
            v = track[field].values
            if track.iloc[0]['Born','Meta']:            
                # If the cell was born, then go and grab the mother cell's division frame for iloc[0]
                motherID = int(track.iloc[0]['Mother'])
                assert(not np.isnan(motherID))
                mother_div_frame = int(track.reset_index().iloc[0]['Frame'] - 1)
                # Check that this frame exists in mother dataframe
                if mother_div_frame in tracks[motherID].index:
                    mother_value = tracks[motherID].loc[mother_div_frame,motherID][field].values
                else:
                    mother_value = np.nan
                v = np.insert(v,0,mother_value)
                
            else:
                # Pad with nan otherwise
                v = np.insert(v,0,np.nan)
                
            v = v[:len(v)-1]
            
            _track[f'{field} at {num_frames_lookback} frame prior'] = v
        
        df_lookback.append(_track)
        
    return pd.concat(df_lookback)

# Grab all fields that has match
measurement_fields = all_df.xs('Measurement',axis=1,level=1).columns
fields2lookback = set(reduce(list.__add__, [measurement_fields[measurement_fields.str.contains(query)].tolist()
    for query in ['Nuclear volume','Height to BM','adjac','Cell volume','Num basal neighbors']] ))

df_lookback = lookback(tracks,fields2lookback)

new_cols = pd.DataFrame()
new_cols['Name'] = df_lookback.columns
new_cols['Metadata'] = 'Measurement'
df_lookback.columns = pd.MultiIndex.from_frame(new_cols)

all_df = all_df.join(df_lookback,how='outer')

all_df.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.pkl'))







