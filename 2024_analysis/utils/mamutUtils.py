#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 18:01:35 2022

@author: xies
"""

import pandas as pd
from os import path
import xml.etree.ElementTree as ET

def sort_links_by_time(links,spots):
    for idx,link in links.iterrows():
        
        source = spots[spots['ID'] == link['SPOT_SOURCE_ID']].iloc[0]
        target = spots[spots['ID'] == link['SPOT_TARGET_ID']].iloc[0]
        
        if source['FRAME'] >= target['FRAME']:
            # Need to swap the source and target
            links.at[idx,'SPOT_SOURCE_ID'] = target['ID']
            links.at[idx,'SPOT_TARGET_ID'] = source['ID']
            
    return links

def trace_single_generation(root,spots,linkages):
    # Trace from root and check outgoing links
    
    # First need to check that this is either the lineage root or a newly born cell
    assert(root['Spot_N_links_N_incoming_links'] == 0 or root['Phase'] == 'Birth' or root['Phase'] == 'Visible birth')
    
    current_spot = root
    this_cell = []
    TERMINATE = False
    
    while not TERMINATE:
        
        this_cell.append(current_spot)
        next_spotID = linkages[linkages['SourceID'] == int(current_spot['SpotID'])]['TargetID'].values
        
        # Seems like there are 'self' linkages?
        next_spotID = next_spotID[next_spotID != current_spot['SpotID']]
        
        if len(next_spotID) == 0:
            TERMINATE = True
            
        elif len(next_spotID) == 1:
            
            assert(len(next_spotID) == 1)
            next_spotID = next_spotID[0]
            current_spot = spots[spots['SpotID'] == next_spotID].iloc[0]
            
        elif len(next_spotID) == 2:
            TERMINATE = True
            this_cell[-1]['Phase'] = 'Division'
            # print(current_spot)
            # Find the two next spots
            assert(len(next_spotID) == 2)
            this_cell[-1]['DaughterID_1'] = next_spotID[0]
            this_cell[-1]['DaughterID_2'] = next_spotID[1]

    this_cell = pd.DataFrame(this_cell)
    return this_cell

def trace_lineage(lineage_root,_spots,_linkage_table, lineageID, trackID = 1):

    all_cells_in_lineage = []
    roots2trace = [lineage_root]
    
    while len(roots2trace) > 0:
        
        this_cell = trace_single_generation(roots2trace.pop(),_spots,_linkage_table)
        this_cell['TrackID'] = trackID
        this_cell['LineageID'] = lineageID
        all_cells_in_lineage.append(this_cell)
        
        if this_cell.iloc[-1]['Phase'] == 'Division':
            daughter1 = _spots[_spots['SpotID'] == this_cell.iloc[-1]['DaughterID_1']].iloc[0]
            daughter2 = _spots[_spots['SpotID'] == this_cell.iloc[-1]['DaughterID_2']].iloc[0]
            # Only overwrite if not 'Visible birth':
            if daughter1['Phase'] != 'Visible birth':
                daughter1['Phase'] = 'Birth'
            if daughter2['Phase'] != 'Visible birth':
                daughter2['Phase'] = 'Birth'
            daughter1['MotherID'] = trackID
            daughter2['MotherID'] = trackID
            roots2trace.append(daughter1)
            roots2trace.append(daughter2)
            
        trackID += 1
    return all_cells_in_lineage


def load_mamut_and_prune_for_complete_cycles(dirname,subdir_str='MaMuT/'):

    raw_spots = pd.read_csv(path.join(dirname,subdir_str,'spots.csv'),skiprows=[1,2,3],header=0)
    raw_spots = raw_spots[raw_spots['TRACK_ID'] != 'None']
    raw_spots['TRACK_ID'] = raw_spots['TRACK_ID'].astype(int)
    raw_links = pd.read_csv(path.join(dirname,subdir_str,'linkage.csv'),skiprows=[1,2,3],header=0)
    raw_tracks = pd.read_csv(path.join(dirname,subdir_str,'tracks.csv'),skiprows=[1,2,3],header=0)
    
    # Do pre-filtering
    # Filter out tracks with fewer than 2 splits (i.e. no complete cell cycles)
    cycling_tracks = raw_tracks[raw_tracks['NUMBER_SPLITS'] > 1]
    
    raw_links = sort_links_by_time(raw_links,raw_spots)
    
    # Separate all spots/linkage into lists of dfs pertaining to only a single track
    cycling_spots = []
    cycling_links = []
    for tID in cycling_tracks['TRACK_ID'].values:
        cycling_spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
        cycling_links.append(raw_links[raw_links['TRACK_ID'] == tID])
    
    return cycling_tracks, cycling_links, cycling_spots

def load_mamut_densely(dirname,subdir_str='MaMuT/'):
    # Load all MaMuT tracks without any pruning

    raw_spots = pd.read_csv(path.join(dirname,subdir_str,'spots.csv'),skiprows=[1,2,3],header=0)
    raw_spots = raw_spots[raw_spots['TRACK_ID'] != 'None']
    raw_spots['TRACK_ID'] = raw_spots['TRACK_ID'].astype(int)
    raw_links = pd.read_csv(path.join(dirname,subdir_str,'linkage.csv'),skiprows=[1,2,3],header=0)
    raw_tracks = pd.read_csv(path.join(dirname,subdir_str,'tracks.csv'),skiprows=[1,2,3],header=0)
    
    
    raw_links = sort_links_by_time(raw_links,raw_spots)
    
    # Separate all spots/linkage into lists of dfs pertaining to only a single track
    _spots = []
    _links = []
    for tID in raw_tracks['TRACK_ID'].values:
        _spots.append(raw_spots[raw_spots['TRACK_ID'] == tID])
        _links.append(raw_links[raw_links['TRACK_ID'] == tID])
    
    return raw_tracks, _links, _spots


def construct_data_frame_dense(_tracks,_links,_spots):
    # For each track, extract individual time series into a dataframe and return a list of all tracks
    # @todo: think about how to deal with multiple generations
    # Will return multiple generations if there are any?
    
    tracks = []
    
    for i in range(len(_tracks)):
        
        link = _links[i]
        spots_ = _spots[i]
        
        spots = pd.DataFrame()
        # Construct a cleaned-up dataframe
        spots['LABEL'] = spots_['LABEL']
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
        spots = spots.sort_values('Frame')
        
        # For each spot, figure out how many incoming links + outgoing links (i.e. mother/daughters)
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
        
        spots2trace = [spots.head(1)]
        while len(spots2trace) > 0:
        
            # Pop from list
            spot = spots2trace.pop()
            print(f'Tracing from {spot.ID.values}')
            track = [spot]
            while not spot.iloc[0]['Division'] and not spot.iloc[0]['Terminus']:
                # Trace the linkages
                next_spot = spots[spots.ID == spot['Left'].iloc[0]]
                track.append(next_spot)
                spot = next_spot
            if spot.iloc[0]['Terminus']:
                # Tracking ended, no further daughters
                tracks_.append(pd.concat(track))
            if spot.iloc[0]['Division']:
                # If we found a division, then this is a complete cell cycle
                tracks_.append(pd.concat(track))
                daughter_a = spots[spots['ID'] == spot.iloc[0]['Left']]
                daughter_b = spots[spots['ID'] == spot.iloc[0]['Right']]
                # Append the new daughters into the todo list
                spots2trace.extend([daughter_a, daughter_b])
        
        tracks.extend(tracks_)

    return tracks



def construct_data_frame_complete_cycles(cycling_tracks,cycling_links, cycling_spots):
    # Traverse linkage trees to prune out the complete cycles
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

    return tracks

