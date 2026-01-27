#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:08:18 2025

@author: xies
"""

# Core libraries
import numpy as np
from skimage import io
import pandas as pd
import matplotlib.pylab as plt

# General utils
from tqdm import tqdm
from os import path
import os
# from basicUtils import nonans

dx = 0.25
dz = 1


# Filenames
# name,dirname = 'R1','/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
name,dirname = 'R2','/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_pca.csv'),
                     header=[0,1],index_col=[0,1])

all_df['Region'] = name

# fields2estimate = ['Nuclear volume','Cell volume']
# df_by_frame = [x for _,x in all_df.groupby('Frame')]

# for t,this_frame in enumerate(df_by_frame):
#     for field in fields2estimate:
#         meas_type = this_frame[field].columns.values
#         basals = this_frame[this_frame['Cell type','Meta'] == 'Basal']
#         this_frame[f'{field} standard'] = \
#             this_frame[field,meas_type] / basals[field,meas_type].dropna().mean()
#     df_by_frame[t] = this_frame

# all_df = pd.concat(df_by_frame,ignore_index=True).set_index(['Frame','TrackID'])

#%% Manuallly annotate cell cycle transisions

# Visible mitosis -- exclude from volume measurements
all_df['Cell cycle transition','Meta'] = 'NA'

#W-R1
if dirname == '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/':
    all_df.loc[(1,47),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(3,251),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(3,902),('Cell cycle transition','Meta')] = 'Mitosis'
    # all_df.loc[(5,1206),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(6,278),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(6,989),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(6,623),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(8,703),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(9,210),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(9,453),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(9,841),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,402),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,40),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,1005),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(11,867),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(11,908),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(11,676),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(12,892),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(12,523),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(13,916),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(13,449),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(14,579),('Cell cycle transition','Meta')] = 'Mitosis'

#W-R2
elif dirname == '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/':
    all_df.loc[(0,1159),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(1,734),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(3,1080),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(2,307),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(2,208),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(3,404),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(4,888),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(4,94),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(5,356),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(6,326),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(7,941),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(7,845),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(8,676),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,827),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,870),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,693),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,966),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,805),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(10,870),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(11,1154),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(12,487),('Cell cycle transition','Meta')] = 'Mitosis'
    all_df.loc[(12,419),('Cell cycle transition','Meta')] = 'Mitosis'

from measurements import map_tzyx_to_labels

all_df['Cell cycle phase','Meta'] = 'G1' # or G1
all_df = all_df.reset_index().set_index('TrackID')

tracked_nuc = io.imread(path.join(dirname,'Mastodon/tracked_nuc.tif'))
filename = path.join(dirname,'Mastodon/NA.csv')
if path.exists(filename):
    na_points = pd.read_csv(filename,index_col=0)
    na_points = na_points.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
    na_points = map_tzyx_to_labels(na_points, tracked_nuc)
    for _,row in na_points.iterrows():
        all_df.loc[row['label'],('Cell cycle phase','Meta')] = 'NA'

sg2_points = pd.read_csv(path.join(dirname,'Mastodon/SG2.csv'),index_col=0)
sg2_points = sg2_points.rename(columns={'axis-0':'T','axis-1':'Z','axis-2':'Y','axis-3':'X'})
sg2_points = map_tzyx_to_labels(sg2_points, tracked_nuc)

all_df = all_df.reset_index().set_index(['Frame','TrackID'])
for _,row in sg2_points.iterrows():
    all_df.loc[(row['T'],row['label']),('Cell cycle phase','Meta')] = 'SG2'

#%% Annotate growth rate / mother / daughter fates

from measurements import detect_missing_frame_and_fill, get_interpolated_curve, get_exponential_growth_rate

tracks = [x for _,x in all_df.reset_index().groupby('TrackID')]

for i,track in tqdm(enumerate(tracks)):

    track = detect_missing_frame_and_fill(track)

    # Birth lineage annotation
    track['Birth frame','Meta'] = False
    if len(track['Mother','Meta'].dropna()) > 0:
        track['Age','Measurement time'] = \
            (track['Time','Measurement time'] - track.iloc[0]['Time','Measurement time'])
        track['Born','Meta'] = True
        track.loc[ track.Frame.argmin(),('Cell cycle transition','Meta') ] = 'Born'
        track.loc[0,('Birth frame','Meta')] = True # Guaranteed via the detect-missing-fill func that index is sorted by frame
    else:
        track['Age','Measurement time'] = np.nan
        track['Born','Meta'] = False

    if len(track['Daughter a','Meta'].dropna()) > 0:
        track['Will divide','Meta'] = True
        track['Divide next frame','Meta'] = False
        track.loc[track.Frame.argmax(),('Divide next frame','Meta')] = True
    else:
        track['Will divide','Meta'] = False
        track['Divide next frame','Meta'] = False
        
    track['Differentiated','Meta'] = (track['Cell type','Meta'] == 'Suprabasal') \
        | (track['Cell type','Meta'] == 'Right before cornified')
    track['Will differentiate','Meta'] = False
    track['Delaminate next frame','Meta'] = False

    # FUCCI intensity
    # track['Cell cycle phase'] = 'NA'
    # track['Max FUCCI frame'] = np.nan
    # if track.iloc[0]['Will differentiate']:
    #     track['Cell cycle phase'] = 'G1'
    # if track.iloc[0]['Will divide']:
    #     max_fucci_idx = track['Mean FUCCI intensity'].argmax()
    #     if max_fucci_idx > -1 and len(track) > 2:
    #         track.loc[max_fucci_idx,'Max FUCCI frame'] = track.loc[max_fucci_idx,'Frame']
    #         prev_frames = track['Frame'] <= max_fucci_idx
    #         track.loc[prev_frames, 'Cell cycle phase'] = 'G1'
    #         after_frames = track['Frame'] > max_fucci_idx
    #         track.loc[after_frames, 'Cell cycle phase'] = 'SG2'

    # Differentiation
    if np.any(track['Differentiated','Meta']) and np.any(track['Cell type','Meta'] == 'Basal'):
        track['Will differentiate','Meta'] = True
        track.loc[track[track['Cell type','Meta'] == 'Basal'].index[-1],'Delaminate next frame'] = True

        diff_frame = np.where(track['Differentiated','Meta'])[0][0]
        diff_idx = track.iloc[diff_frame].name
        track['Time to differentiation','Measurement time'] = \
            track['Time','Measurement time'] - track.iloc[diff_frame]['Time','Measurement time']
        track['Keep until first differentiation','Meta'] = False
        if (track['Cell type','Meta'] == 'Basal').sum() > 0:
            track.loc[ (track['Cell type','Meta'] == 'Basal'),('Keep until first differentiation','Meta')] = True
            track.loc[diff_idx, ('Keep until first differentiation','Meta')] = True

    try:
        #@todo: rewrite fcn with column levels
        track = get_interpolated_curve(track, field=('Nuclear volume','Measurement nuclear shape'))
    except UserWarning:
        breakpoint()
        
    track = get_exponential_growth_rate(track, field=('Nuclear volume','Measurement nuclear shape'))
    Ig1 = track['Cell cycle phase','Meta'] == 'G1'
    track = get_exponential_growth_rate(track, 
                                        field=('Nuclear volume','Measurement nuclear shape'), filtered={'G1 only':Ig1})

    track = get_interpolated_curve(track, field=('Cell volume','Measurement cell shape'))
    track = get_exponential_growth_rate(track, field=('Cell volume','Measurement cell shape'))
    track = get_exponential_growth_rate(track, field=('Cell volume','Measurement cell shape'), filtered={'G1 only':Ig1})

    track = get_interpolated_curve(track, field=('Mean FUCCI intensity','Measurement FUCCI'))
    track = get_interpolated_curve(track, field=('Total H2B intensity','Measurement H2B'))

    track = get_interpolated_curve(track, field=('Basal area','Measurement cell shape'))
    track = get_interpolated_curve(track, field=('Apical area','Measurement cell shape'))

    track = get_interpolated_curve(track,
                                   field=('Collagen alignment to basal footprint','Measurement collagen'))
    track = get_interpolated_curve(track,
                                   field=('Collagen intensity','Measurement collagen'))

    tracks[i] = track
    # new_tracks.append(track)

all_df = pd.concat(tracks,ignore_index=True).set_index(['Frame','TrackID'])

all_df['Fate known','Meta'] = \
    all_df['Will differentiate','Meta'] | all_df['Will divide','Meta']

all_df.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics.pkl'))

#%% Separate meta and measurement fields and save

# meta_cols = ['TrackID','LineageID','Left','Right','Division','Terminus',
#              'Mother','Sister','Daughter a','Daughter b','Cell type','Reviewed',
#              'Cutoff','Complete cycle','Will divide','Divide next frame',
#              'Differentiated','Will differentiate','Border','Fate known',
#              'Delaminate next frame','Keep until first differentiation',
#              'Born','Birth frame','Cell cycle transition',
#              'Max FUCCI frame','Cell cycle phase']

# Imeta = np.isin(all_df.columns,meta_cols)
# metadata_index = {True:'Meta',False:'Measurement'}
# metadata_index = [metadata_index[x] for x in Imeta]

# new_cols = pd.DataFrame()
# new_cols['Name'] = all_df.columns
# new_cols['Metadata'] = metadata_index

# all_df.columns = pd.MultiIndex.from_frame(new_cols)

# all_df.to_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics.pkl'))
