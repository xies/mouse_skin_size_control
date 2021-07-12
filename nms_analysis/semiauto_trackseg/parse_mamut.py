#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:58:19 2021

Parses .csv output of Mamut and prunes out complete cell cycles

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

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1/'

#%% Load CSV mamut exports

raw_spots = pd.read_csv(path.join(dirname,'spots.csv'))
raw_links = pd.read_csv(path.join(dirname,'linkage.csv'))
raw_tracks = pd.read_csv(path.join(dirname,'tracks.csv'))

# Do pre-filtering
# Filter out tracks with fewer than 2 splits (i.e. no complete cell cycles)

cycling_tracks = raw_tracks[raw_tracks['NUMBER_SPLITS'] > 1]
num_tracks = len(cycling_tracks)

# Separate all spots/linkage into lists of dfs pertaining to only a single track
cycling_spots = []
cycling_links = []
for tID in cycling_tracks['TRACK_ID'].values:
    cycling_spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
    cycling_links.append(raw_links[raw_links['TRACK_ID'] == tID])
    
#%% Traverse linkage trees to prune out the complete cycles

#NB: Spots and links are chronologically sorted (descending)

tracks = []
for i in range(len(cycling_tracks)):
    
    link = cycling_links[i]
    spots_ = cycling_spots[i]
        
    spots = pd.DataFrame()
    # Construct a cleaned-up dataframe
    spots['ID'] = spots_['ID']
    spots['X'] = spots_['POSITION_X']
    spots['Y'] = spots_['POSITION_Y']
    spots['Z'] = spots_['POSITION_Z']
    spots['Frame'] = spots_['POSITION_T']
    spots['TrackID'] = spots_['TRACK_ID']
    spots['Left'] = None
    spots['Right'] = None
    spots['Division'] = False
    spots['Terminus'] = False
    
    
    # Build a daughter(s) tree into the spots dataframe
    
    for idx,spot in spots.iterrows():
        links_from_this_spot = link[link['SPOT_SOURCE_ID'] == spot.ID]
        
        if len(links_from_this_spot) == 1:
            # No division
            spots.at[idx,'Left'] = links_from_this_spot.iloc[0]['SPOT_TARGET_ID']
        elif len(links_from_this_spot) == 2:
            # There are two daughters
            spots.at[idx,'Left'] = links_from_this_spot.iloc[0]['SPOT_TARGET_ID']
            spots.at[idx,'Right'] = links_from_this_spot.iloc[1]['SPOT_TARGET_ID']
            spots.at[idx,'Division'] = True
        elif len(links_from_this_spot) == 0:
            spots.at[idx,'Terminus'] = True
    
    # From first cell division, initiate a Track object and traverse until we hit either: 1) division or 2) terminus
    # If Division, complete current Track and try to initiate another Track object
    # If Terminus, delete current Track object
    
    divisions = spots[ spots['Division'] == True ]
    first_division = divisions.sort_values('Frame').iloc[0]
    
    daughter_a = spots[spots['ID'] == first_division['Left']]
    daughter_b = spots[spots['ID'] == first_division['Right']]
    
    tracks_ = []
    
    spots2trace = [daughter_a, daughter_b]
    while len(spots2trace) > 0:
    
        # Pop from list
        spot = spots2trace.pop()
        print(f'Tracing from {spot.ID}')
        track = [spot]
        while not spot.iloc[0]['Division'] and not spot.iloc[0]['Terminus']:
            # Trace the linkages
            next_spot = spots[spots.ID == spot['Left'].iloc[0]]
            track.append(next_spot)
            spot = next_spot
        if spot.iloc[0]['Division']:
            # If we found a division, then this is a complete cell cycle
            tracks_.append(pd.concat(track))
            daughter_a = spots[spots['ID'] == spot.iloc[0]['Left']]
            daughter_b = spots[spots['ID'] == spot.iloc[0]['Right']]
            # Append the new daughters into the todo list
            spots2trace.extend([daughter_a, daughter_b])
    
    tracks.extend(tracks_)


#%% Export the coordinates of the completed cell cycles (as pickle)
with open(path.join(dirname,'complete_cycles.pkl'),'wb') as file:
    pkl.dump(tracks,file)

