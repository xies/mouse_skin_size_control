#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:26 2024

@author: xies
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from os import path
from tqdm import tqdm
from mamutUtils import trace_lineage

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'

#%%

filename = path.join(dirname,'dataset_deconv_Pos2_reviewedMimi-mamut.xml')

root = ET.parse(filename).getroot()

# Convert spots into a dataframe
_spots = pd.DataFrame([pd.Series(s.attrib) for s in root.iter('Spot')]).astype(float)
_spots = _spots.rename(columns={'ID':'SpotID'
                                ,'POSITION_X':'X','POSITION_Y':'Y','POSITION_Z':'Z'
                                ,'Frame':'Frame'})
_spots['TrackID'] = np.nan
_spots['LineageID'] = np.nan
_spots['Phase'] = 'NA'
_spots['MotherID'] = np.nan
_spots['DaughterID_1'] = np.nan
_spots['DaughterID_2'] = np.nan

# Annotage G1/S and visible birth frames since we can't infer from lineage topology
g1s_spots = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/g1s-Spot.csv'),skiprows=[1,2])
_spots.loc[np.isin(_spots['SpotID'],g1s_spots['ID']),'Phase'] = 'G1S'
g1s_spots = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/birth-Spot.csv'),skiprows=[1,2])
_spots.loc[np.isin(_spots['SpotID'],g1s_spots['ID']),'Phase'] = 'Visible birth'

#%%

'''
Follow linkages from the root, check number of outgoing links every time.
If outgoing links == 0: terminate but don't mark as DIVISION
If outgoing links == 1: follow as normal
If outgoing links == 2: terminate current cell as DIVISION, initiate two daughter cells from BIRTH
Need to mark: cell cycle phase: If terminus, mark division; if newly born, mark birth

NB: These could be overwritten with manual cell cycle annotations from mastodon export

'''

_lineages = [t for t in root.iter('Track')]

# Keep track of lineage + track numbers
lineageID = 1
trackID = 1

all_lineages = []
for t in tqdm(_lineages):
    
    
    lineageID += 1
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
    this_lineage = trace_lineage(lineage_root, _spots, _linkage_table, lineageID= lineageID, trackID = trackID)

    # Filter for cells with manual 'G1S' annotations ONLY
    this_lineage = [t for t in this_lineage if 
                    (t['Phase'] == 'G1S').sum() > 0]
    all_lineages.extend(this_lineage)
    
    if len(all_lineages) > 0:
        trackID = all_lineages[-1].iloc[0]['TrackID'] + 1

#%% Go through each track and mark G1 v. SG2

for i,t in enumerate(all_lineages):
    
    t = t.sort_values('FRAME').reset_index()
    t = t.drop(columns=['index'])
    idx_nan = np.where(t['Phase'] == 'NA')[0]
    first_g1s_idx = t.index[np.where(t['Phase'] == 'G1S')[0][0]]
    t.loc[idx_nan[idx_nan < first_g1s_idx],'Phase'] = 'G1'
    
    last_g1s_idx = t.index[np.where(t['Phase'] == 'G1S')[0][-1]]
    t.loc[idx_nan[idx_nan > last_g1s_idx],'Phase'] = 'SG2'
    
    all_lineages[i] = t

pd.concat(all_lineages).to_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'))


