#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:56:15 2025

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import seaborn as sb
from os import path

import pickle as pkl

from mamutUtils import load_mamut_xml_densely, construct_data_frame_dense

#%% Export the coordinates of the completed cell cycles (as pickle)

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R2/'

all_tracks = []
_tracks, _spots = load_mamut_xml_densely(path.join(dirname,'Mastodon/R2-mamut.xml'))
tracks = construct_data_frame_dense(_tracks, _spots)

all_tracks.append(tracks)

#%% Annotations

# Merge with manual tags using Spot.csv files
spot_table = pd.read_csv(path.join(dirname,'Mastodon/R2-spots-Spot.csv'),
                         header=[0,1,2],index_col=0)

# Only select the labels that matter:
spot_table = spot_table.loc[:,['Cell type','Reviewed by','ID']].convert_dtypes(float)
spot_table.columns = spot_table.columns.droplevel(2)
spot_table.columns = spot_table.columns.map('_'.join)
spot_table = spot_table.rename(columns={'ID_Unnamed: 1_level_1':'ID'})

# Reverse hot-1 encoding
spot_table['Type'] = ''
for col in spot_table.columns[spot_table.columns.str.startswith('Cell type')]:
    spot_table.loc[spot_table[col] == 1, 'Type'] = col.split('_')[1]

spot_table = spot_table.rename(columns={'Type':'Cell type'})

#%%

for track in tracks:
    track['Cell type'] = 'NA'
    track['Reviewed by'] = False
    
    for idx,spot in track.iterrows():
        _spot = spot_table[spot_table['ID'] == float(spot.ID)]
        track.loc[idx,'Cell type'] = _spot['Cell type'].values
        if _spot['Reviewed by_Mimi'].values == 1:
            track.loc[idx,'Reviewed'] = True
                
#%%

border = 4 #micron from edge
maxT = 14
maxXY = 115

def is_on_border(track,border,maxXY):
    track['X'] = track['X'].astype(float)
    track['Y'] = track['Y'].astype(float)
    Ix = (track['X'] < border) | (track['X'] > maxXY-border)
    Iy = (track['Y'] < border) | (track['Y'] > maxXY-border)
    return np.any(Ix | Iy)

for track in tracks:
    
    # Annotate border
    track['Border'] = is_on_border(track,border,maxXY)
    # Annotate terminal (ends at T = 14)
    track['Cutoff'] = float(track.iloc[-1]['Frame']) == maxT
    # Annotate complete cell cycle
    track['Complete cycle'] = ~np.isnan(float(track.iloc[0]['Daughter a'])) \
        & ~np.isnan(float(track.iloc[0]['Mother']))

with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)

#%%

# plt.boxplot([wtlength,rbkolength],labels=['WT','RB-KO'])
# plt.ylabel('Cell cycle length (h)')

# plt.figure()

# plt.hist(wtlength,12,histtype='step');plt.hist(rbkolength,12,histtype='step')
# plt.legend(['WT','RB-KO'])

# plt.xlabel('Cell cycle length (h)')

