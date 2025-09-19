#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 11:25:04 2025

@author: xies
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# General utils
from tqdm import tqdm
from os import path


import ipyvolume as ipv
from sklearn import preprocessing, decomposition
from ppca import PPCA   
from measurements import get_prev_or_next_frame, scale_by_region

dx = 0.25
dz = 1

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

#%% Annotate the cell cycle times in cells where information is available

all_df['Cell cycle duration','Measurement'] = np.nan
all_df['G1 duration','Measurement'] = np.nan
all_df['SG2 duration','Measurement'] = np.nan
for trackID,track in all_tracks.items():
    if track.iloc[0]['Complete cycle','Meta']:
        all_df.loc[zip(track['Frame'].values,track['TrackID'].values),
                   ('Cell cycle duration','Measurement')] = len(track)*12
    if np.any(track['Cell cycle phase','Meta'] == 'SG2'):
        all_df.loc[zip(track['Frame'].values,track['TrackID'].values),
                   ('G1 duration','Measurement')] = (track['Cell cycle phase','Meta'] == 'G1').sum()*12
        if track.iloc[0]['Complete cycle','Meta']:
            all_df.loc[zip(track['Frame'].values,track['TrackID'].values),
                       ('SG2 duration','Measurement')] = (track['Cell cycle phase','Meta'] == 'SG2').sum()*12
        # track['G1 duration'] = (track['Cell cycle phase','Meta'] == 'G1').sum()*12
        # track['SG2 duration'] = (track['Cell cycle phase','Meta'] == 'SG2').sum()*12
        

#%% Isolate time points of interest. Don't need to know the fate for now for the PCA part.

print(f'Number of total time points: {len(all_df)}')

df = all_df[ ~all_df['Border','Meta'].astype(bool)]
basals = df[ df['Cell type','Meta'] == 'Basal']
basals = basals[ ~basals['Border','Meta'].astype(bool)]
# basals = basals[ basals['Frac of neighbors are border','Meta'] < 0.2]
print(f'Number of non-border basal cells: {len(basals)}')

births = basals[basals['Birth frame','Meta']]
births_raw = births.copy()
print(f'Number of non-border births (fate known or unknown): {len(births)}')

divisions = basals[basals[('Divide next frame','Meta')]].copy()
divisions = divisions[~divisions['Border','Meta'].astype(bool)]
divisions = divisions.reset_index()
print(f'Number of non-border mother divisions (daughter fate known or unkonwn): {len(divisions)}')

prev_div_frame = [get_prev_or_next_frame(basals,f,direction='prev') for _,f in divisions.iterrows()]
prev_div_frame = pd.concat(prev_div_frame,axis=1,ignore_index=False).T
for col in df.columns:
    prev_div_frame[col] = prev_div_frame[col].astype(df[col].dtype)
prev_div_frame = prev_div_frame[~prev_div_frame['Border','Meta'].astype(bool)]
prev_div_frame = prev_div_frame.reset_index().rename(columns={'level_0':'Frame','level_1':'TrackID'}).set_index('TrackID')
print(f'Number of 12h prior to divisions: {len(prev_div_frame)}')

prev2_div_frame = [get_prev_or_next_frame(all_df,f,direction='prev', increment=2) for _,f in divisions.iterrows()]
prev2_div_frame = pd.concat(prev2_div_frame,axis=1).T
prev2_div_frame = prev2_div_frame[~prev2_div_frame['Border','Meta'].astype(bool)]
for col in df.columns:
    prev2_div_frame[col] = prev2_div_frame[col].astype(df[col].dtype)
print(f'Number of 24h prior to divisions: {len(prev2_div_frame)}')

prev3_div_frame = [get_prev_or_next_frame(all_df,f,direction='prev', increment=3) for _,f in divisions.iterrows()]
prev3_div_frame = pd.concat(prev3_div_frame,axis=1).T
prev3_div_frame = prev3_div_frame[~prev3_div_frame['Border','Meta'].astype(bool)]
for col in df.columns:
    prev3_div_frame[col] = prev3_div_frame[col].astype(df[col].dtype)
print(f'Number of 36h prior to divisions: {len(prev3_div_frame)}')

prev4_div_frame = [get_prev_or_next_frame(all_df,f,direction='prev', increment=4) for _,f in divisions.iterrows()]
prev4_div_frame = pd.concat(prev4_div_frame,axis=1).T
prev4_div_frame = prev4_div_frame[~prev4_div_frame['Border','Meta'].astype(bool)]
for col in df.columns:
    prev4_div_frame[col] = prev4_div_frame[col].astype(df[col].dtype)
print(f'Number of 48h prior to divisions: {len(prev4_div_frame)}')

divisions = divisions.set_index(['Frame','TrackID'])
prev_div_frame = prev_div_frame.reset_index().set_index(['Frame','TrackID'])

#%% Save these time points

model_dir = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Lineage models/'
dataset_dir = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/Lineage models/Dataset pickles'

all_df.to_pickle(path.join(dataset_dir,'all_df.pkl'))
basals.to_pickle(path.join(dataset_dir,'basals.pkl'))
births.to_pickle(path.join(dataset_dir,'births.pkl'))
divisions.to_pickle(path.join(dataset_dir,'divisions.pkl'))
prev_div_frame.to_pickle(path.join(dataset_dir,'divisions_12h.pkl'))
prev2_div_frame.to_pickle(path.join(dataset_dir,'divisions_24h.pkl'))
prev3_div_frame.to_pickle(path.join(dataset_dir,'divisions_36h.pkl'))
prev4_div_frame.to_pickle(path.join(dataset_dir,'divisions_48h.pkl'))

#%% PPCA different time points


features2drop_from_pca = ['X','Y','Z','X-pixels','Y-pixels','X-cyto','Y-cyto','Z-cyto']
def get_ppca_component_transform(dataset:pd.DataFrame,
                                 n_comp:int=100,
                                 features2drop=None,
                                 name:str=''):
    
    dataset = dataset.drop(columns=features2drop)
    
    X = scale_by_region(df)
    feature_names_with_nan = X.drop(columns='Region').columns
    X = X.drop(columns='Region')
    nonan_idx = np.where(~np.isnan(X).all(axis=0))[0]
    feature_names = feature_names_with_nan[nonan_idx]
    
    ppca = PPCA()
    ppca.fit(X.values, d=n_comp, verbose=False) #NB: will silently drop np.all NaN columns
    
    components = pd.DataFrame(ppca.C, index=feature_names)
    X_transformed = pd.DataFrame(ppca.transform(),index=df.index)
    X_transformed.columns = pd.MultiIndex.from_tuples(
        [(f'PC{i}{name}','PCA') for i in range(n_comp)])

    return ppca, components, X_transformed
    

time_points = {'all_df':all_df,'births':births,'basals':basals,
               'divisions':divisions,'divisions_12h':prev_div_frame,
               'divisions_24h':prev2_div_frame,
               'divisions_36h':prev3_div_frame,'divisions_48h':prev4_div_frame,}

for name,df in tqdm(time_points.items()):
    
    _df = df.xs('Measurement',axis=1,level=1)
    pca, components, transformed = get_ppca_component_transform(_df,
                                                                features2drop= features2drop_from_pca,
                                                                name=f'_{name}')
    
    pca.save(path.join(model_dir,f'Probabilistic PCA/pca_{name}'))
    components.to_pickle(path.join(model_dir,f'Probabilistic PCA/{name}/components.pkl'))
    # Join PCA components with the metadata
    metadata = df.xs('Meta',axis=1,level=1)
    metadata['Region'] = df['Region']
    metadata.columns = pd.MultiIndex.from_tuples([(c,'Meta') for c in metadata.columns])
    df_transformed = pd.concat((transformed,metadata),axis=1)

    df_transformed.to_pickle(path.join(model_dir,f'Probabilistic PCA/{name}/transformed.pkl'))
    
    # transformed


