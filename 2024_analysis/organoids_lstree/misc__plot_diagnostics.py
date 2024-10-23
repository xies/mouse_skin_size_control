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

df_combined = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)

tracks = {trackIDs:t for trackIDs,t in df_combined.groupby('trackID')}
trackIDs = list(tracks.keys())

#%% Single variables

i = 27

t = tracks[trackIDs[i]]
g1 = t[~t['Auto phase']]
sg2 = t[t['Auto phase']]

plt.subplot(2,1,1)
plt.plot(g1['Age'],g1['Normalized Cdt1 intensity'])
plt.plot(sg2['Age'],sg2['Normalized Cdt1 intensity'])

plt.subplot(2,1,2)
plt.plot(g1['Age'],g1['Change in Cdt1'])
plt.plot(sg2['Age'],sg2['Change in Cdt1'])


#%% Plot logistic curves

# Truncate
tracks_g1s = {}
for trackID,track in tracks.items():
    I = np.where(track['Auto phase'])[0]
    if I.sum()>0:
        tracks_g1s[trackID] = track.iloc[:I[0]+1]

df_g1s = pd.concat(tracks_g1s,ignore_index=True).dropna()
sb.regplot(df_g1s,x='Normalized Cdt1 intensity',y='Auto phase',y_jitter=0.1,logistic=True)

