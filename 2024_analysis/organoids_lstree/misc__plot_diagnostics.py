#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:04:28 2024

@author: xies
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from os import path

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
df5 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
df5['organoidID'] = 5
df5 = df5[ (df5['trackID'] !=77) & (df5['trackID'] != 120) & (df5['trackID'] != 88)]

# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
# df2 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
# df2['organoidID'] = 2
# df2 = df2[ (df2['trackID'] !=53) & (df2['trackID'] != 6)]

# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 6_2um/'
# df6 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_features.csv'),index_col=0)
# df6['organoidID'] = 6

# df = pd.concat((df5,df2),ignore_index=True)
df = df5
df['organoidID_trackID'] = df['organoidID'].astype(str) + '_' + df['trackID'].astype(str)
df = df.dropna(subset='trackID')

tracks = {trackID:t for trackID,t in df.groupby('organoidID_trackID')}

# First, drop everything but first G1/S frame
# g1s_tracks = {}
# for trackID,track in tracks.items():
#     I = track['Auto phase']
#     if I.sum() > 0:
#         first_g1s_idx = np.where(I)[0][0]
#         g1s_tracks[trackID] = track.iloc[0:first_g1s_idx+1]
        
# g1s = pd.concat(tracks, ignore_index=True)

#%% Single variables

trackIDs = list(tracks.keys())
trackOI = trackIDs[45]
print(f'TrackID = {trackOI}')
# trackOI = 46
t = tracks[ trackOI ]

# plt.plot(t.Frame,t['Nuclear volume'])


g1 = t[t['Phase'] != 'Visible birth']
sg2 = t[t['Phase'] == 'SG2']

# plt.subplot(2,1,1)
plt.plot(g1['Frame'],g1['Normalized Cdt1 intensity'])
plt.plot(sg2['Frame'],sg2['Normalized Cdt1 intensity'])

plt.subplot(2,1,2)
plt.plot(g1['Frame'],g1['Nuclear volume'])
plt.plot(sg2['Frame'],sg2['Nuclear volume'])

#%% Plot logistic curves

# df_g1s_balanced = np.

# Truncate
tracks_g1s = {}
for trackID,track in tracks.items():
    I = np.where(track['Auto phase'])[0]
    if I.sum()>0:
        tracks_g1s[trackID] = track.iloc[:I[0]+1]

df_g1s = pd.concat(tracks_g1s,ignore_index=True)

# sb.regplot(df_g1s,x='Normalized Cdt1 intensity',y='Auto phase',y_jitter=0.1,logistic=True)

plt.figure()
sb.regplot(df_g1s,x='Nuclear volume',y='Auto phase',y_jitter=0.1,logistic=True)

