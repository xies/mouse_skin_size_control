#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:45:37 2022

Parses .csv output of Mamut and quantify growth rates

Exports (pickle) as a list of dataframes, each corresponding to a complete cycle (from birth to division).
Exported fields:
    SpotID  X    Y    Z    T     Left child      Right child    Division(flag)     Terminus (flag)  Source (flag)

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

num_tracks = len(raw_tracks)

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
tmp_spots = []
tmp_links = []
for tID in raw_tracks['TRACK_ID'].values:
    tmp_spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
    tmp_links.append(raw_links[raw_links['TRACK_ID'] == tID])


# tmp_tracks = []
spots = []

# Parse each lineage tree and identify SOURCE and TERMINUS spots
for i in range(num_tracks):
    
    link = tmp_links[i]
    this_spots_ = tmp_spots[i].sort_values(by='POSITION_T')
        
    this_spots = pd.DataFrame()
    # Construct a cleaned-up dataframe
    this_spots['ID'] = this_spots_['ID']
    this_spots['X'] = this_spots_['POSITION_X']
    this_spots['Y'] = this_spots_['POSITION_Y']
    this_spots['Z'] = this_spots_['POSITION_Z']
    this_spots['Frame'] = this_spots_['POSITION_T']
    this_spots['TrackID'] = this_spots_['TRACK_ID']
    this_spots['Left'] = None
    this_spots['Right'] = None
    this_spots['Division'] = False
    this_spots['Terminus'] = False
    this_spots['Source'] = False
    
    # The first spots (all spots should be sorted by frame) and all subsequent result of splits are 'sources'
    this_spots.at[this_spots.index[0],'Source'] = True

    # Build a daughter(s) tree into the spots dataframe
    # Notes: COULD build this on the first pass-through, but it's probably safer to build a temporary database and further split
    for idx,spot in this_spots.iterrows():
        links_from_this_spot = link[link['SPOT_SOURCE_ID'] == spot.ID]
        
        if len(links_from_this_spot) == 1:
            # No division
            this_spots.at[idx,'Left'] = links_from_this_spot.iloc[0]['SPOT_TARGET_ID']
        elif len(links_from_this_spot) == 2:
            # There are two daughters
            this_spots.at[idx,'Left'] = links_from_this_spot.iloc[0]['SPOT_TARGET_ID']
            this_spots.at[idx,'Right'] = links_from_this_spot.iloc[1]['SPOT_TARGET_ID']
            this_spots.at[idx,'Division'] = True
            
            this_spots.at[ this_spots[this_spots['ID'] == this_spots.loc[idx]['Left']].index[0],'Source'] = True
            this_spots.at[ this_spots[this_spots['ID'] == this_spots.loc[idx]['Right']].index[0],'Source'] = True
            
            
        elif len(links_from_this_spot) == 0:
            this_spots.at[idx,'Terminus'] = True
    
    spots.append(this_spots)


#%% Separate mother / daughter tracks

tracks = []

for i,this_spots in enumerate(spots):
    
    link = tmp_links[i]
    assert(sum(this_spots['Source']) > 0)
    
    # If only a single SOURCE then it's a linear track
    if sum(this_spots['Source']) == 1:
        tracks.append(this_spots)
        
    # If there are splits
    else:
        sources = this_spots[this_spots['Source'] & ~this_spots['Terminus'] & ~this_spots['Division']]
        for _,source in sources.iterrows():

            links_from_this_spot = link[link['SPOT_SOURCE_ID'] == source.ID]
            assert(len(links_from_this_spot) == 1) # should only be one outgoing track
            
            TERMINATED = False
            track_ = []
            track_.append(source)
            while not TERMINATED:
                
                next_spot = this_spots[ this_spots['ID'] == links_from_this_spot['SPOT_TARGET_ID'].iloc[0]].iloc[0]
                source = next_spot
                links_from_this_spot = link[link['SPOT_SOURCE_ID'] == source.ID]
                
                track_.append(next_spot)
                if next_spot['Terminus']:
                    TERMINATED = True
            
            track_ = pd.DataFrame(track_)
            tracks.append(track_)

#%% Export the coordinates of the completed cell cycles (as pickle)
with open(path.join(dirname,'MaMut/tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)

 