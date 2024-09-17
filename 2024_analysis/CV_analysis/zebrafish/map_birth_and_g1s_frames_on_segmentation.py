#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:07:48 2024

@author: xies
"""

import numpy as np
import pandas as pd
from natsort import natsorted
from os import path
from glob import glob
from skimage import io
from tqdm import tqdm

import xml.etree.ElementTree as ET

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

# Load MaMuT-XML file
root = ET.parse(path.join(dirname,'Position001_Mastodon/Position_001-mamut.xml')).getroot()

# Parse through all spots
_spots = []
for s in root.iter('Spot'):
    s = pd.Series({'ID':int(s.attrib['ID'])
               ,'X':float(s.attrib['POSITION_X'])
               ,'Y':float(s.attrib['POSITION_Y'])
               ,'Z':float(s.attrib['POSITION_Z'])
               ,'T':float(s.attrib['FRAME']) })
    _spots.append(s)
_spots = pd.DataFrame(_spots)

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

#Write down on the spots df which track it belongs to
for tID, this_sIDs in spotsIDs_belonging_to_track.items():
    for sID in this_sIDs:
        _spots.loc[_spots['ID'] == sID,'TrackID'] = tID

# Transfer the brith+G1S+div annotations
birth_spots = pd.read_csv(path.join(dirname,'Position001_Mastodon/birth/birth-Spot.csv'),skiprows=[1,2])
_spots['Birth'] = False
for _,spot in birth_spots.iterrows():
    _spots.loc[_spots['ID'] == spot['ID'],'Birth'] = True
    
g1s_spots = pd.read_csv(path.join(dirname,'Position001_Mastodon/g1s/g1s-Spot.csv'),skiprows=[1,2])
_spots['G1S'] = False
for _,spot in g1s_spots.iterrows():
    _spots.loc[_spots['ID'] == spot['ID'],'G1S'] = True     

div_spots = pd.read_csv(path.join(dirname,'Position001_Mastodon/division/division-Spot.csv'),skiprows=[1,2])
_spots['G1S'] = False
for _,spot in div_spots.iterrows():
    _spots.loc[_spots['ID'] == spot['ID'],'Division'] = True            

# Group out each track and sort
birth_spots = _spots[_spots['Birth'] == True]
birth_spots = [t.sort_values('T') for _,t in birth_spots.groupby('TrackID')]

g1s_spots = _spots[_spots['G1S'] == True]
g1s_spots = [t.sort_values('T') for _,t in g1s_spots.groupby('TrackID')]

div_spots = _spots[_spots['Division'] == True]
div_spots = [t.sort_values('T') for _,t in div_spots.groupby('TrackID')]

#%% Load segmentation files

radius = 5

XX= 1024
ZZ = 38
TT = 60

mCh_files = natsorted(glob(path.join(dirname,'Position001_Mastodon/predicted_labels/*mch**labels.tif')))
ven_files = natsorted(glob(path.join(dirname,'Position001_Mastodon/predicted_labels/*ven**labels.tif')))

# Use 1-indexed trackIDs for now?
current_trackID = 0

birth_img = np.zeros((TT,ZZ,XX,XX),np.uint16)
for this_cell in tqdm(birth_spots):
    # tID = int(this_cell.iloc[0]['TrackID'])
    current_trackID += 1
    
    for _,this_frame in this_cell.iterrows():
        
        x = int(np.round(this_frame['X']))
        y = int(np.round(this_frame['Y']))
        z = int(np.round(this_frame['Z']))
        t = int(np.round(this_frame['T']))
        im = io.imread(ven_files[t])
        label = im[z,y,x]
        if label == 0:
            im = io.imread(mCh_files[t])
            label = im[z,y,x]
        if label > 0:
            mask = im == label
            birth_img[t,mask] = current_trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            # print(f'Time {t} -- {i}: {ID}')
            y_low = max(0,y - radius); y_high = min(XX,y + radius)
            x_low = max(0,x - radius); x_high = min(XX,x + radius)
            z_low = max(0,z - radius); z_high = min(ZZ,z + radius)
            birth_img[t,z_low:z_high, y_low:y_high, x_low:x_high] = current_trackID

io.imsave(path.join(dirname,'Position001_Mastodon/birth/birth_predicted.tif'), birth_img)

#%%

current_trackID = 0

g1s_img = np.zeros((TT,ZZ,XX,XX),np.uint16)
for this_cell in tqdm(g1s_spots):
    # tID = int(this_cell.iloc[0]['TrackID'])
    current_trackID += 1
    
    
    for _,this_frame in this_cell.iterrows():
        
        x = int(np.round(this_frame['X']))
        y = int(np.round(this_frame['Y']))
        z = int(np.round(this_frame['Z']))
        t = int(np.round(this_frame['T']))
        im = io.imread(mCh_files[t])
        label = im[z,y,x]
        if label == 0:
            im = io.imread(ven_files[t])
            label = im[z,y,x]
        if label > 0:
            mask = im == label
            g1s_img[t,mask] = current_trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            # print(f'Time {t} -- {i}: {ID}')
            y_low = max(0,y - radius); y_high = min(XX,y + radius)
            x_low = max(0,x - radius); x_high = min(XX,x + radius)
            z_low = max(0,z - radius); z_high = min(ZZ,z + radius)
            g1s_img[t,z_low:z_high, y_low:y_high, x_low:x_high] = current_trackID

io.imsave(path.join(dirname,'Position001_Mastodon/g1s/g1s_predicted.tif'), g1s_img)

#%% division

current_trackID = 0

div_img = np.zeros((TT,ZZ,XX,XX),np.uint16)
for this_cell in tqdm(div_spots):
    # tID = int(this_cell.iloc[0]['TrackID'])
    current_trackID += 1
    
    
    for _,this_frame in this_cell.iterrows():
        
        x = int(np.round(this_frame['X']))
        y = int(np.round(this_frame['Y']))
        z = int(np.round(this_frame['Z']))
        t = int(np.round(this_frame['T']))
        im = io.imread(mCh_files[t])
        label = im[z,y,x]
        if label == 0:
            im = io.imread(ven_files[t])
            label = im[z,y,x]
        if label > 0:
            mask = im == label
            div_img[t,mask] = current_trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            # print(f'Time {t} -- {i}: {ID}')
            y_low = max(0,y - radius); y_high = min(XX,y + radius)
            x_low = max(0,x - radius); x_high = min(XX,x + radius)
            z_low = max(0,z - radius); z_high = min(ZZ,z + radius)
            div_img[t,z_low:z_high, y_low:y_high, x_low:x_high] = current_trackID

io.imsave(path.join(dirname,'Position001_Mastodon/division/div_predicted.tif'), div_img)

