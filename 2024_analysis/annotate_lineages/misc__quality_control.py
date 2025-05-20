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

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics.csv'),
                     index_col=['Frame','TrackID'])

all_df = all_df[ ~all_df['Border']]
tracks = [t for _,t in all_df.reset_index().groupby('TrackID') if np.all(~t.Border)]

#%%

from measurements import plot_track
from cycler import cycler
cc = ['r','g','b','k','c','m','y','orange','brown','gray']
c = {True:'b',False:'r'}

start = 20

plt.figure()
for i,track in enumerate(tracks[start:start+1000]):
    if track.Border.sum() == 0:
        if track.iloc[0]['Will differentiate']:
            plt.subplot(2,1,1)
        else:
            plt.subplot(2,1,2)
            
        plot_track(track, field='Nuclear volume',time='Age',alpha=0.1,
                   color=c[track.iloc[0]['Will differentiate']])
        
# plt.legend()

