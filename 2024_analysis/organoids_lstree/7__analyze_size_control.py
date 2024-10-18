#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:31:40 2024

@author: xies
"""

import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
dx = 0.26
dz = 2

df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features.csv'),index_col=0)

#%%

# Filter out all non-tracked cells
tracks = {trackID:t for trackID,t in df.groupby('trackID')}

summary = pd.DataFrame()
# Extract birth, G1S, division
for trackID, track in tracks.items():
    
    # Skip tetraploids
    if trackID == 77 or trackID == 120:
        continue
    
    # Birth
    # I = track['Phase'] == 'Visible birth'
    summary.loc[trackID,'Birth volume'] = track.iloc[:2]['Nuclear volume (sm)'].mean()
    #first G1S
    I = track['Phase'] == 'G1S'
    summary.loc[trackID,'G1S volume'] = track.iloc[np.where(I)[0][:2]]['Nuclear volume (sm)'].mean()
    # Div
    I = track['Phase'] == 'Division'
    summary.loc[trackID,'Division volume'] = track.iloc[np.where(I)[0][:2]]['Nuclear volume (sm)'].mean()
    
    # Find lengths
    summary.loc[trackID,'Birth time'] = track.iloc[0]['Time']
    I = track['Phase'] == 'G1S'
    if I.sum() > 0:
        summary.loc[trackID,'G1S time'] = track.iloc[np.where(I)[0][0]]['Time'].mean()
    I = track['Phase'] == 'Division'
    if I.sum() > 0:
        summary.loc[trackID,'Division time'] = track.iloc[np.where(I)[0][-1]]['Time'].mean()
    

summary['G1 growth'] = summary['G1S volume'] - summary['Birth volume']
summary['SG2 growth'] = summary['Division volume'] - summary['G1S volume']
summary['Total growth'] = summary['Division volume'] - summary['Birth volume']
summary['G1 length'] = summary['G1S time'] - summary['Birth time']
summary['SG2 length'] = summary['Division time'] - summary['G1S time']
summary['Total length'] = summary['Division time'] - summary['Birth time']

plt.figure()
plt.subplot(2,1,1)
plt.scatter(summary['Birth volume'],summary['G1 growth'])
plt.ylabel('G1 growth (fL)')
plt.subplot(2,1,2)
plt.scatter(summary['Birth volume'],summary['G1 length'])
plt.ylabel('G1 length (min)')

plt.figure()
plt.subplot(2,1,1)
plt.scatter(summary['G1S volume'],summary['SG2 growth'])
plt.ylabel('SG12 growth (fL)'); plt.xlabel('G1S size')
plt.subplot(2,1,2)
plt.scatter(summary['G1S volume'],summary['SG2 length'])
plt.ylabel('G2 length (min)')


#%% load old organoid data

dirnames = ['/Users/xies/Onedrive - Stanford/In vitro/mIOs/Light sheet movies/20200303_194709_09',
            '/Users/xies/Onedrive - Stanford/In vitro/mIOs/Light sheet movies/20200306_165005_01']
homeo = pd.concat([pd.read_csv(path.join(d,'size_control.csv')) for d in dirnames])


plt.scatter(homeo['Birth volume']*1.5**2,homeo['G1 growth']*1.5**2)
plt.scatter(summary['Birth volume'],summary['G1 growth'])

#%%