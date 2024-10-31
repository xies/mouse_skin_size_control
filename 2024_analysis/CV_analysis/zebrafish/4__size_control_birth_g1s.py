#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:10:32 2024

@author: xies
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from os import path
from natsort import natsorted
from glob import glob
from tqdm import tqdm
from mamutUtils import trace_lineage
from skimage import io, measure

import matplotlib.pyplot as plt

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/Position001_Mastodon/'

#%%

filename = path.join(dirname,'Position001_mastodon_new.xml')

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
g1s_spots = pd.read_csv(path.join(dirname,'birth_to_g1s_tracking/g1s-Spot.csv'),skiprows=[1,2])
_spots.loc[np.isin(_spots['SpotID'],g1s_spots['ID']),'Phase'] = 'G1S'
g1s_spots = pd.read_csv(path.join(dirname,'birth_to_g1s_tracking/birth-Spot.csv'),skiprows=[1,2])
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
lineageID = 0

all_lineages = []
for t in tqdm(_lineages):
    
    trackID = len(all_lineages) + 1
    
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

    # Filter for cells with manual 'G1S' annotations AND 'Birth'
    this_lineage = [t for t in this_lineage if 
                    (t['Phase'] == 'G1S').sum() > 0 and (t['Phase'] == 'Visible birth').sum() > 0]
    all_lineages.extend(this_lineage)


pd.concat(all_lineages).to_csv(path.join(dirname,'birth_to_g1s_tracking/filtered_tracks.csv'))


#%% Load segmentation files

def fill_in_with_cube(label,img,x,y,z,t, radius=5):
    [TT,ZZ,YY,ZZ] = img.shape
    y_low = max(0,y - radius); y_high = min(XX,y + radius)
    x_low = max(0,x - radius); x_high = min(XX,x + radius)
    z_low = max(0,z - radius); z_high = min(ZZ,z + radius)
    img[t,z_low:z_high, y_low:y_high, x_low:x_high] = label
    return img

XX= 1024
ZZ = 38
TT = 60

mCh_files = natsorted(glob(path.join(dirname,'predicted_labels/*mch**labels.tif')))
ven_files = natsorted(glob(path.join(dirname,'predicted_labels/*ven**labels.tif')))

# load manual segs also
birth_manual = io.imread(path.join(dirname,'birth_to_g1s_tracking/birth_manual.tif'))
g1s_manual = io.imread(path.join(dirname,'birth_to_g1s_tracking/g1s_manual.tif'))


# Use 1-indexed trackIDs for now?
current_trackID = 0

birth_img = np.zeros((TT,ZZ,XX,XX),np.uint16)
g1s_img = np.zeros((TT,ZZ,XX,XX),np.uint16)
for this_cell in tqdm(all_lineages):
    current_trackID += 1
    
    birth_frames = this_cell[this_cell['Phase'] == 'Visible birth']
    for _,this_frame in birth_frames.iterrows():
        [x,y,z,t] = this_frame[['X','Y','Z','FRAME']].astype(int)
        im = birth_manual[t,...]
        label = im[z,y,x]
        if label == 0:
            im = io.imread(ven_files[t])
            label = im[z,y,x]
        if label > 0:
            mask = im == label
            birth_img[t,mask] = current_trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            birth_img = fill_in_with_cube(current_trackID,birth_img,x,y,z,t)
            
    g1s_frames = this_cell[this_cell['Phase'] == 'G1S']
    for _,this_frame in g1s_frames.iterrows():
        [x,y,z,t] = this_frame[['X','Y','Z','FRAME']].astype(int)
        im = g1s_manual[t,...]
        label = im[z,y,x]
        if label == 0:
            im = io.imread(mCh_files[t])
            label = im[z,y,x]
        if label > 0:
            mask = im == label
            g1s_img[t,mask] = current_trackID
        else:
            # Create a 'cube' around spots that are missing segmentations
            g1s_img = fill_in_with_cube(current_trackID,g1s_img,x,y,z,t)

io.imsave(path.join(dirname,'birth_to_g1s_tracking/birth_tracked.tif'), birth_img)
io.imsave(path.join(dirname,'birth_to_g1s_tracking/g1s_tracked.tif'), g1s_img)

#%%

dx = 0.6
dz = 1.8

dropped_indices = np.array([9,16,21,27,31,39])

# Load file manifest
manifest = pd.read_pickle(path.join(
    path.split(path.split(dirname)[0])[0],'Position001_manifest.pkl'))
# def correct_for_dropped_frames(t, dropped_indices):
#     num_correct_for = (t >= dropped_indices).sum()
#     return t + num_correct_for

birth_img = io.imread(path.join(dirname,'birth_to_g1s_tracking/birth_manual.tif'))
g1s_img = io.imread(path.join(dirname,'birth_to_g1s_tracking/g1s_manual.tif'))

df = []
for t in tqdm(range(TT)):
    births = pd.DataFrame(measure.regionprops_table(birth_img[t,...],properties=['area','label']))
    g1s = pd.DataFrame(measure.regionprops_table(g1s_img[t,...],properties=['area','label']))
    births = births.rename(columns={'area':'Birth size'})
    births['Birth frame'] = manifest.loc[t,'Elapsed']
    g1s = g1s.rename(columns={'area':'G1S size'})
    g1s['G1 frame'] = births['Birth frame'] = manifest.loc[t,'Elapsed']
    
    df.extend([births,g1s])

df = pd.concat(df)
df.to_csv(path.join(dirname,'birth_to_g1s_tracking/size_control_summary.csv'))


#%%

df = pd.read_csv(path.join(dirname,'birth_to_g1s_tracking/size_control_summary.csv'),index_col=0)

mean_bsize = df.groupby('label')['Birth size'].apply(np.nanmean)
mean_g1ssize = df.groupby('label')['G1S size'].apply(np.nanmean)
mean_g1growth = mean_g1ssize - mean_bsize

g1_duration = df.groupby('label').min()['G1 frame'] - df.groupby('label').min()['Birth frame']

plt.figure()
plt.scatter(mean_bsize*dx**2*dz,g1_duration.dt.total_seconds()/3600)
plt.xlabel('Nuclear size at birth (fL)')
plt.ylabel('G1 duration (h)')
plt.figure()
plt.scatter(mean_bsize*dx**2*dz,mean_g1growth)
plt.xlabel('Nuclear size at birth (fL)')
plt.ylabel('G1 growth (fL)')








