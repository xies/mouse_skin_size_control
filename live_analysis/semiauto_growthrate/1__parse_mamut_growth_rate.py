#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:45:37 2022

Parses .csv output of Mamut and quantify growth rates

Exports (pickle) as a list of dataframes, each corresponding to a complete cycle (from birth to division).
Exported fields:
    SpotID  X    Y    Z    T     Left child      Right child    Division(flag)     Terminus (flag)

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import seaborn as sb
from os import path

import pickle as pkl

# Avoid parsing XML
# import xml.etree.ElementTree as ET

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1'

#%% Load CSV mamut exports

raw_spots = pd.read_csv(path.join(dirname,'MaMut/spots.csv'),skiprows=[1,2,3],header=0)
raw_spots = raw_spots[raw_spots['TRACK_ID'] != 'None']
raw_spots['TRACK_ID'] = raw_spots['TRACK_ID'].astype(int)
raw_links = pd.read_csv(path.join(dirname,'MaMuT/linkage.csv'),skiprows=[1,2,3],header=0)
raw_tracks = pd.read_csv(path.join(dirname,'MaMuT/tracks.csv'),skiprows=[1,2,3],header=0)

# Do pre-filtering
# Filter out tracks with fewer than 2 splits (i.e. no complete cell cycles)
cycling_tracks = raw_tracks[raw_tracks['NUMBER_SPLITS'] > 1]
num_tracks = len(cycling_tracks)

def sort_links_by_time(links,spots):
    for idx,link in links.iterrows():
        
        source = spots[spots['ID'] == link['SPOT_SOURCE_ID']].iloc[0]
        target = spots[spots['ID'] == link['SPOT_TARGET_ID']].iloc[0]
        
        if source['FRAME'] >= target['FRAME']:
            # Need to swap the source and target
            links.at[idx,'SPOT_SOURCE_ID'] = target['ID']
            links.at[idx,'SPOT_TARGET_ID'] = source['ID']
            
    return links

raw_links = sort_links_by_time(raw_links,raw_spots)


# Separate all spots/linkage into lists of dfs pertaining to only a single track
cycling_spots = []
cycling_links = []
for tID in cycling_tracks['TRACK_ID'].values:
    cycling_spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
    cycling_links.append(raw_links[raw_links['TRACK_ID'] == tID])


#%% Export the coordinates of the completed cell cycles (as pickle)
with open(path.join(dirname,'MaMuT/complete_cycles.pkl'),'wb') as file:
    pkl.dump(tracks,file)

