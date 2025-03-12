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

dirname ='/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

all_tracks = []
_tracks, _spots = load_mamut_xml_densely(dirname,subdir_str='Mastodon')
tracks = construct_data_frame_dense(_tracks, _spots)

with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)

all_tracks.append(tracks)

#%% Annotations

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
    # Annotate differentiated
    track['Differentiated'] = track.iloc[-1]['Terminus'] & ~track.iloc[-1]['Cutoff']


with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)


#%%

# plt.boxplot([wtlength,rbkolength],labels=['WT','RB-KO'])
# plt.ylabel('Cell cycle length (h)')

# plt.figure()

# plt.hist(wtlength,12,histtype='step');plt.hist(rbkolength,12,histtype='step')
# plt.legend(['WT','RB-KO'])

# plt.xlabel('Cell cycle length (h)')

