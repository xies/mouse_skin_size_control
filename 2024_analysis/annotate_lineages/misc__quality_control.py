#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:02:07 2025

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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints.csv'),
                     index_col=['Frame','TrackID'])
all_tracks = {trackID:t for trackID,t in all_df.reset_index().groupby('TrackID')}

all_df = all_df[ ~all_df['Border']]
all_df = all_df[ all_df['Fate known']]
tracks = {trackID:t for trackID,t in all_df.reset_index().groupby('TrackID') if np.all(~t.Border)}

#%%

from measurements import plot_track
# from cycler import cycler
cc = ['r','g','b','k','c','m','y','orange','brown','gray']
c = {True:'b',False:'r'}

keys = list(tracks.keys())
# keys = [431]

div_count = 0; delam_count = 0
plt.figure()
keysplotted = []

div_mothers = []
diff_mothers = []

for i,track in enumerate([tracks[k] for k in keys]):
    
    t = track
    # t = track[(track['Cell type'] == 'Basal')]
    if len(t) == 0:
        continue
    
    if not np.any(t.iloc[0]['Border']) and np.all(t['Born']):
        mother = all_tracks[ int(t.iloc[-1]['Mother']) ]
        
        if len(t) > 0:
            
            if t.Border.sum() == 0:
                # if t.iloc[0]['Will differentiate']:
                if np.any(t['Will divide']):
                    plt.subplot(1,2,1)
                    div_count += 1
                    div_mothers.append(mother)
                    
                elif np.any(t['Will differentiate']):
                    plt.subplot(1,2,2)
                    delam_count += 1
                    diff_mothers.append(mother)
                else:
                    continue
                    
                plot_track(mother, field='Nuclear volume smoothed',time='Age',alpha=0.2,
                           color=c[t.iloc[0]['Will differentiate']])
                keysplotted.append(t.iloc[0].TrackID)
                
                # plt.ylim([0,400])
                plt.xlim([0,120])

            
plt.subplot(1,2,1); plt.title('Will divide')
plt.subplot(1,2,2); plt.title('Will differentiate')
# plt.xlabel('Time since birth (h)')
plt.ylabel('Nuclear volume (fL)')
# plt.legend()

#%%

all_df = pd.concat(tracks,ignore_index=True)
before_division = all_df[all_df['Divide next frame']]
before_delamination = all_df[all_df['Delaminate next frame']]



