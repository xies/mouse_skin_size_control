#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:03:48 2022

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io
import seaborn as sb
from os import path
from glob import glob

import pickle as pkl


dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M2 RB-KO/R1'
# dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/05-03-2021 Rb-fl/M1 WT/R1'
# dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/10-20-2021/WT/R1/'

dx = 0.2920097

#%% Collate G1 annotations

# with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'wb') as file:
#     pkl.dump(tracks,file)

with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
    tracks = pkl.load(file)

g1_annotation = pd.read_excel(path.join(dirname,'manual_tracking','g1_annotations.xlsx'))
g1_annotation['G1 exit'] = g1_annotation['G1 exit frame'].astype(float)

for track in tracks:
    
    corrID = track.iloc[0]['CorrID']
    if ~np.isnan(corrID):
    
        # Find same corrID in annotation
        track['G1 exit'] = g1_annotation[g1_annotation['CorrID'] == corrID]['G1 exit'].values[0]-1
        
        
        frames = track['Frame']
        first_frame = frames.min()
        track['t'] = (track['Frame'] - first_frame) / 2.0 # half-day
        
        g1_exit_frame = track.iloc[0]['G1 exit']
        track['Phase'] = 'NA'
        if ~np.isnan(g1_exit_frame):
            # Annotate cell cycle phase on time-series
            track['Phase'] = 'G1'
            track.at[track['Frame'] >= track.iloc[0]['G1 exit'],'Phase'] = 'SG2'
    

ts = pd.concat(tracks)

# Save to the manual folder    
with open(path.join(dirname,'final_trackseg/complete_cycles_final.pkl'),'wb') as file:
    pkl.dump(tracks,file)
    
#%% Start cell-centric dataframe with cell-level data:

df = pd.DataFrame()

for track in tracks:
    
    corrID = track.iloc[0]['CorrID']
    
    if ~np.isnan(corrID) and len(track) > 1:
    
        cycle_length = track['Age'].max()
        birth_size = track.iloc[0]['Volume'] * dx**2
        birth_frame = track.iloc[0]['Frame']
        division_size = track.iloc[-2]['Volume'] * dx**2
        division_growth = division_size - birth_size
    
        g1_size = np.nan
        g1_length = np.nan
        g1_growth = np.nan
        
        # Check to see if there is any G1 annotation
        if ~np.isnan( track.iloc[0]['G1 exit'] ):
            I_g1 = track['G1 exit'].iloc[0] == track['Frame']
            if np.any(I_g1):
                g1_length = track[I_g1]['Age'].values[0]
                g1_size = track[ I_g1 ]['Volume'].values[0] * dx**2
                g1_growth = g1_size - birth_size
            
        df = df.append( pd.Series(name=corrID,data = {'CorrID':corrID,'Cycle length':cycle_length,
                                             'Birth size':birth_size,'Birth frame':birth_frame,
                                             'G1 exit size':g1_size,'G1 length':g1_length,
                                             'G1 growth':g1_growth,'Division size':division_size,
                                             'Total growth':division_growth}).to_frame().T )
        
df['SG2 length'] = df['Cycle length'] - df['G1 length']
df['SG2 growth'] = df['Total growth'] - df['G1 growth']

# Save
df.to_csv(path.join(dirname,'final_trackseg','cell_dataframe.pkl'))


