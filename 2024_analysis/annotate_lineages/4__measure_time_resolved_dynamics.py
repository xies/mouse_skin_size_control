#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:08:18 2025

@author: xies
"""

# Core libraries
import numpy as np
# from skimage import io
import pandas as pd
import matplotlib.pylab as plt

# General utils
from tqdm import tqdm
from os import path
# from basicUtils import nonans

dx = 0.25
dz = 1

# Filenames
dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),index_col=[0,1]).reset_index()

fields2estimate = ['Nuclear volume','Cell volume']
df_by_frame = [x for _,x in all_df.groupby('Frame')]

for t,this_frame in enumerate(df_by_frame):
    for field in fields2estimate:
        basals = this_frame[this_frame['Cell type'] == 'Basal']
        this_frame[f'{field} standard'] = \
            this_frame[field] / basals[field].dropna().mean()
    df_by_frame[t] = this_frame
    
all_df = pd.concat(df_by_frame,ignore_index=True).set_index(['Frame','TrackID'])

#%% Annotate growth rate / mother / daughter

from measurements import detect_missing_frame_and_fill, get_interpolated_curve, get_exponential_growth_rate

tracks = [x for _,x in all_df.reset_index().groupby('TrackID')]
fields2estimate = ['Nuclear volume','Nuclear volume standard']

for i,track in tqdm(enumerate(tracks)):
    
    track = detect_missing_frame_and_fill(track)
    
    track['Birth frame'] = False
    if len(track['Mother'].dropna()) > 0:
        track['Age'] = (track['Time'] - track.iloc[0]['Time'])
        track['Born'] = True
        track.loc[0,'Birth frame'] = True # Guaranteed via the detect-missing-fill func that index is sorted by frame
    else:
        track['Age'] = np.nan
        track['Born'] = False
    
    if len(track['Daughter a'].dropna()) > 0:
        track['Will divide'] = True
        track['Divide next frame'] = False
        track.loc[track.Frame.argmax(),('Divide next frame')] = True
    else:
        track['Will divide'] = False
        track['Divide next frame'] = False
    
    track['Differentiated'] = (track['Cell type'] == 'Suprabasal') \
        | (track['Cell type'] == 'Right before cornified')
    track['Will differentiate'] = False
    track['Delaminate next frame'] = False
    
    if np.any(track['Differentiated']) and np.any(track['Cell type'] == 'Basal'):
        track['Will differentiate'] = True
        track.loc[track[track['Cell type'] == 'Basal'].index[-1],'Delaminate next frame'] = True
        
        diff_frame = np.where(track['Differentiated'])[0][0]
        diff_idx = track.iloc[diff_frame].name
        track['Time to differentiation'] = track['Time'] - track.iloc[diff_frame]['Time']
        track['Keep until first differentiation'] = False
        if (track['Cell type'] == 'Basal').sum() > 0:
            track.loc[ track['Cell type'] == 'Basal','Keep until first differentiation'] = True
            track.loc[diff_idx, 'Keep until first differentiation'] = True
    
    track = get_interpolated_curve(track, field='Nuclear volume')
    track = get_exponential_growth_rate(track, field='Nuclear volume')
    
    track = get_interpolated_curve(track, field='Cell volume')
    track = get_exponential_growth_rate(track, field='Cell volume')
    
    track = get_interpolated_curve(track, field='Mean FUCCI intensity')
    track = get_interpolated_curve(track, field='Total H2B intensity')
    
    track = get_interpolated_curve(track, field='Basal area')
    track = get_interpolated_curve(track, field='Apical area')
    
    track = get_interpolated_curve(track, field='Basal alignment')
    track = get_interpolated_curve(track, field='Collagen intensity')
    
    tracks[i] = track
    # new_tracks.append(track)

all_df = pd.concat(tracks,ignore_index=True).set_index(['Frame','TrackID'])

all_df['Fate known'] = all_df['Will differentiate'] | all_df['Will divide']

#%%

# Separate meta and measurement fields
meta_cols = ['TrackID','LineageID','Left','Right','Division','Terminus',
             'Mother','Sister','Daughter a','Daughter b','Cell type','Reviewed',
             'Cutoff','Complete cycle','Will divide','Divide next frame',
             'Differentiated','Will differentiate','Border','Fate known',
             'Delaminate next frame','Keep until first differentiation',
             'Born','Birth frame']

Imeta = np.isin(all_df.columns,meta_cols)
metadata_index = {True:'Meta',False:'Measurement'}
metadata_index = [metadata_index[x] for x in Imeta]

new_cols = pd.DataFrame()
new_cols['Name'] = all_df.columns
new_cols['Metadata'] = metadata_index

all_df.columns = pd.MultiIndex.from_frame(new_cols)

all_df.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics.pkl'))


