#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:42:26 2024

@author: xies
"""

import xml.etree.ElementTree as ET


filename = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/dataset_deconv_Pos5_reviewedMimi-mamut.xml'




<Spot ID="34406" name="34406" POSITION_X="68.23838378907038" POSITION_Y="56.44155976465579" POSITION_Z="47.157406584924104" FRAME="86" POSITION_T="86.0" QUALITY="-1.0" VISIBILITY="1" RADIUS="2.9601262620089046" Spot_radius="2.960126262008904" Spot_N_links_N_outgoing_links="0.0" Spot_N_links_Spot_N_links="1.0" Spot_N_links_N_incoming_links="1.0" Spot_position_X="68.23838378907038" Spot_position_Y="56.44155976465579" Spot_position_Z="47.157406584924104" Spot_frame="86.0" />

<Track name="13" TRACK_ID="11">
  <Edge SPOT_SOURCE_ID="11" SPOT_TARGET_ID="87" Link_displacement="0.7353910524340038" Link_target_IDs_Source_spot_id="11.0" Link_target_IDs_Target_spot_id="87.0" Link_velocity="0.7353910524340038" Link_delta_T="1.0" />

root = ET.parse(filename).getroot()

_tracks = {}
spotsIDs_belonging_to_track = {}
for track in root.iter('Track'):
    # NB: Edge object not guaranteed to be 'chronological'
    _this_edges = []
    for e in track.iter('Edge'):
        e = pd.Series({'SourceID':int(e.attrib['SPOT_SOURCE_ID'])
            ,'TargetID':int(e.attrib['SPOT_TARGET_ID']) })
        _this_edges.append(e)
    _this_edges = pd.DataFrame(_this_edges)
    # _this_edges['TrackID'] = track.attrib['TRACK_ID']
    spotsIDs_belonging_to_track[int(track.attrib['TRACK_ID'])] = set([*_this_edges['SourceID'],*_this_edges['TargetID']])
    
    _tracks[int(track.attrib['TRACK_ID'])] = _this_edges
    
