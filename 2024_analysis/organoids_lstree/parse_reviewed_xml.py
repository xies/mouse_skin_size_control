#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:26 2024

@author: xies
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


#%%

filename = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/dataset_deconv_Pos5_reviewedMimi-mamut.xml'

root = ET.parse(filename).getroot()

# Convert spots into a dataframe
_spots = pd.DataFrame([pd.Series(s.attrib) for s in root.iter('Spot')]).astype(float)
_spots = _spots.rename(columns={'ID':'SpotID'
                                ,'POSITION_X':'X','POSITION_X':'Y','POSITION_Z':'Z'
                                ,'Frame':'Frame'})
_spots['TrackID'] = np.nan
_spots['LineageID'] = np.nan
_spots['Phase'] = 'NA'
_spots['MotherID'] = np.nan
_spots['DaughterID_1'] = np.nan
_spots['DaughterID_2'] = np.nan

# Annotage G1/S frames since we can't infer from lineage topology



#%%

'''
Follow linkages from the root, check number of outgoing links every time.
If outgoing links == 0: terminate but don't mark as DIVISION
If outgoing links == 1: follow as normal
If outgoing links == 2: terminate current cell as DIVISION, initiate two daughter cells from BIRTH
Need to mark: cell cycle phase: If terminus, mark division; if newly born, mark birth

NB: These could be overwritten with manual cell cycle annotations from mastodon export

'''


def trace_single_generation(root,spots,linkages):
    # Trace from root and check outgoing links
    
    # First need to check that this is either the lineage root or a newly born cell
    assert(root['Spot_N_links_N_incoming_links'] == 0 or root['Phase'] == 'Birth')
    
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
        else:
            STOP
            break

    this_cell = pd.DataFrame(this_cell)
    return this_cell

def trace_lineage(lineage_root,_spots,_linkage_table, lineageID, trackID = 1):

    all_cells_in_lineage = []
    roots2trace = [lineage_root]
    
    while len(roots2trace) > 0:
        
        this_cell = trace_single_generation(roots2trace.pop(),_spots,_linkage_table)
        this_cell['TrackID'] = trackID
        all_cells_in_lineage.append(this_cell)
        
        if this_cell.iloc[-1]['Phase'] == 'Division':
            daughter1 = _spots[_spots['SpotID'] == this_cell.iloc[-1]['DaughterID_1']].iloc[0]
            daughter2 = _spots[_spots['SpotID'] == this_cell.iloc[-1]['DaughterID_2']].iloc[0]
            daughter1['Phase'] = 'Birth'
            daughter2['Phase'] = 'Birth'
            daughter1['MotherID'] = trackID
            daughter2['MotherID'] = trackID
            roots2trace.append(daughter1)
            roots2trace.append(daughter2)
            
        trackID += 1
    return all_cells_in_lineage

_lineages = [t for t in root.iter('Track')]

lineageID = 1

all_lineages = []
for t in _lineages:
    
    _linkage_table = []
    for e in t.iter('Edge'):
        e = pd.Series({'SourceID':int(e.attrib['SPOT_SOURCE_ID'])
            ,'TargetID':int(e.attrib['SPOT_TARGET_ID']) })
        _linkage_table.append(e)
    _linkage_table = pd.DataFrame(_linkage_table)
    
    spotsIDs_belonging_to_track = set([*_linkage_table['SourceID'],*_linkage_table['TargetID']])
    
    spots_in_track = _spots[np.isin(list(_spots['SpotID'].values),list(spotsIDs_belonging_to_track))].sort_values('Spot_frame')
    
    # Only the 'root' of the lineage will have NO incoming linkages
    lineage_root = spots_in_track.loc[ spots_in_track['Spot_N_links_N_incoming_links'] == 0 ].iloc[0]
    this_lineage = trace_lineage(lineage_root, _spots, _linkage_table, lineageID= 1, trackID = 1)


    

