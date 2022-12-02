#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'


dx = 0.2920097
# dx = 1


#%% Load and collate manual track+segmentations
# Dictionary of manual segmentation (there should be no first or last time point)
manual_segs = io.imread(path.join(dirname,'manual_tracking/manual_tracking.tif'))

# Resave as 16-bit
# io.imsave(path.join(dirname,'manual_tracking/manual_tracking_8bit.tif'),manual_segs.astype(np.int8))

filename = glob(path.join(dirname,'im_seq/*.tif'))
imstack = list(map(io.imread,filename))

#%% Re-construct tracks with manually fixed tracking/segmentation

trackIDs = np.unique(manual_segs)

tracks = []
for trackID in tqdm(trackIDs[1:]):
    
    track = pd.DataFrame()
    
    mask = manual_segs == trackID
    frames_with_this_track = np.where(np.any(np.any(np.any(mask,axis=1),axis=1), axis=1))[0]
    
    
    for frame in frames_with_this_track:
        
        # Measurements from segmentation/ label iamge
        this_frame = mask[frame,...]
        
        props = measure.regionprops(this_frame*1)
        Z,X,Y = props[0]['Centroid']
        volume = props[0]['Area']
        
        # Measurement from intensity image(s)
        # fucci_this_frame = imstack[frame][:,:,:,0]
        # props = measure.regionprops(this_frame*1, intensity_image = fucci_this_frame)
        # fucci_mean = props[0]['mean_intensity']
        
        track = track.append(pd.Series({'Frame':frame,'X':X,'Y':Y,'Z':Z,'Volume':volume}),
                              ignore_index=True)
        
    track['CorrID'] = trackID
    track['Age'] = (track['Frame'] - track.iloc[0]['Frame'])*12
    
    tracks.append(track)
    print(f'Done with corrID {trackID}')


#@todo: Go back and detect the missing frames and fill-in with NaNs
for i,track in enumerate(tracks):
    
    frames = track['Frame']
    skipped_frames = frames[:-1][np.diff(frames) > 1] # the 'before skip' frame
    for f in skipped_frames:
        
        missing_frame = f+1
        track = track.append(pd.Series({'Frame':missing_frame,'X':np.nan,'Y':np.nan,'Z':np.nan,'Volume':np.nan
                             ,'CorrID' : track.iloc[0].CorrID, 'Age': (missing_frame - track.iloc[0]['Frame']) * 12
                             }),ignore_index=True)
        track = track.sort_values('Frame').reset_index(drop=True)
        tracks[i] = track

# Save to the manual folder    
with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'wb') as file:
    pkl.dump(tracks,file)

# ts = pd.concat(tracks)

#%% Load cell cycle annotations

with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
    tracks = pkl.load(file)

# Load excel annotaitons
filename = path.join(dirname,'cell_cycle_annotations.xlsx')
anno = pd.read_excel(filename,usecols=range(5),index_col=0)

for track in tracks:
    
    track['Phase'] = 'NA'
    track['Birth frame'] = np.nan
    track['Division frame'] = np.nan
    track['S phase entry frame'] = np.nan
    if track.iloc[0].CorrID in anno.index:
        this_anno = anno.loc[track.iloc[0].CorrID]
        
        track['Birth frame'] = this_anno.Birth
        track['Division frame'] = this_anno.Division
        track['S phase entry frame'] = this_anno['S phase entry']
        
        if not np.isnan(this_anno['S phase entry']):
            track.loc[track['Frame'] < this_anno['S phase entry'],'Phase'] = 'G1'
            track.loc[track['Frame'] >= this_anno['S phase entry'],'Phase'] = 'S'
        
        track.loc[track['Frame'] == this_anno.Birth,'Phase'] = 'Birth'
        
        # if not np.isnan(this_anno['Division']):
        #     track.loc[track['Frame'] == this_anno.Division,'Phase'] = 'Division'
        
        track['Mitosis'] = this_anno['Mitosis?'] == 'Yes'
        
# Save to the manual folder    
with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'wb') as file:
    pkl.dump(tracks,file)


# with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
#     tracks = pkl.load(file)

df = []

for track in tracks:
    
    birth_size = np.nan
    div_size = np.nan
    s_size = np.nan
    
    g1_length = np.nan
    total_length = np.nan
    
    birth_frame = track.iloc[0]['Birth frame']
    if not np.isnan(birth_frame):
        birth_size = track[track['Frame'] == birth_frame]['Volume'].values[0]
    else:
        birth_frame = np.nan
        
    div_frame = track.iloc[0]['Birth frame']
    if not np.isnan(div_frame):
        div_size = track[track['Frame'] == div_frame]['Volume'].values[0]
        total_length = track[track['Frame'] == div_frame]['Age'].values[0]
    else:
        div_frame = np.nan
        
    s_frame = track.iloc[0]['S phase entry frame']
    if not np.isnan(s_frame):
        s_size = track[track['Frame'] == s_frame]['Volume'].values[0]
        g1_length = track[track['Frame'] == s_frame]['Age'].values[0]
    else:
        s_frame = np.nan
    
    
    df.append(pd.Series({'CorrID':track.iloc[0].CorrID
                         ,'Birth frame': birth_frame
                         ,'S phase frame': s_frame
                         ,'Birth size': birth_size
                         ,'S phase entry size':s_size
                         ,'G1 length':g1_length
                         ,'Total length':total_length
                         ,'G1 growth':s_size - birth_size
                         ,'Total growth':div_size - birth_size
                         }))
    
df = pd.concat(df,ignore_index=True,axis=1).T

#%% basic plotting



# IDs2plot = np.arange(20,30)

# for corrID in IDs2plot:
#     plt.plot(tracks[corrID]['Frame'],tracks[corrID]['Volume'])
    
    
    


  