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

from mamutUtils import sort_links_by_time, load_mamut_and_prune_for_complete_cycles

#%% Load CSV mamut exports

def sort_links_by_time(links,spots):
    for idx,link in links.iterrows():
        
        source = spots[spots['ID'] == link['SPOT_SOURCE_ID']].iloc[0]
        target = spots[spots['ID'] == link['SPOT_TARGET_ID']].iloc[0]
        
        if source['FRAME'] >= target['FRAME']:
            # Need to swap the source and target
            links.at[idx,'SPOT_SOURCE_ID'] = target['ID']
            links.at[idx,'SPOT_TARGET_ID'] = source['ID']
            
    return links

def load_mamut_and_prune_for_complete_cycles(dirname):
    raw_spots = pd.read_csv(path.join(dirname,'MaMuT/spots.csv'),skiprows=[1,2,3],header=0)

    raw_spots = raw_spots[raw_spots['TRACK_ID'] != 'None']
    raw_spots['TRACK_ID'] = raw_spots['TRACK_ID'].astype(int)
    raw_links = pd.read_csv(path.join(dirname,'MaMuT/linkage.csv'),skiprows=[1,2,3],header=0)
    raw_tracks = pd.read_csv(path.join(dirname,'MaMuT/tracks.csv'),skiprows=[1,2,3],header=0)
    
    # Do pre-filtering
    # Filter out tracks with fewer than 2 splits (i.e. no complete cell cycles)
    cycling_tracks = raw_tracks[raw_tracks['NUMBER_SPLITS'] > 1]
    num_tracks = len(cycling_tracks)
    
    
    raw_links = sort_links_by_time(raw_links,raw_spots)
    
    # Separate all spots/linkage into lists of dfs pertaining to only a single track
    cycling_spots = []
    cycling_links = []
    for tID in cycling_tracks['TRACK_ID'].values:
        cycling_spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
        cycling_links.append(raw_links[raw_links['TRACK_ID'] == tID])
    
    return cycling_tracks, cycling_links, cycling_spots

def construct_data_frame(cycling_tracks,cycling_links, cycling_spots):
    # Traverse linkage trees to prune out the complete cycles
    #NB: Spots and links are chronologically sorted (descending)
    tracks = []
    
    for i in range(len(cycling_tracks)):
        
        link = cycling_links[i]
        spots_ = cycling_spots[i]
            
        spots = pd.DataFrame()
        # Construct a cleaned-up dataframe
        spots['ID'] = spots_['ID']
        spots['Label'] = spots_['LABEL']
        spots['X'] = spots_['POSITION_X']
        spots['Y'] = spots_['POSITION_Y']
        spots['Z'] = spots_['POSITION_Z']
        spots['Frame'] = spots_['POSITION_T']
        spots['MaMuTID'] = spots_['TRACK_ID']
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
            # print(f'Tracing from {spot.ID}')
            track = [spot]

            idx = 0
            while not spot.iloc[0]['Division'] and not spot.iloc[0]['Terminus']:
                idx +=1
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

    return tracks

#%% Export the coordinates of the completed cell cycles (as pickle)

dirnames = []
# dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1')
dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R1')

all_tracks = []
for dirname in dirnames:
    cycling_tracks, cycling_links, cycling_spots = load_mamut_and_prune_for_complete_cycles(dirname)
    tracks = construct_data_frame(cycling_tracks, cycling_links, cycling_spots)

    with open(path.join(dirname,'MaMuT/complete_cycles.pkl'),'wb') as file:
        pkl.dump(tracks,file)

    all_tracks.append(tracks)
    
#%%

rbko = all_tracks[0]
# wt = all_tracks[1]

# wtlength  = (np.array([len(t) for t in wt])* 12)
rbkolength  = (np.array([len(t) for t in rbko])* 12)

# plt.boxplot([wtlength,rbkolength],labels=['WT','RB-KO'])
# plt.ylabel('Cell cycle length (h)')

# plt.figure()

# plt.hist(wtlength,12,histtype='step');plt.hist(rbkolength,12,histtype='step')
# plt.legend(['WT','RB-KO'])

# plt.xlabel('Cell cycle length (h)')

