#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:22:31 2021

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
dx = 0.2920097

# Load final tracks
with open(path.join(dirname,'manual_track','complete_cycles_fixed.pkl'),'rb') as file:
    tracks = pkl.load(file)
    
with open('/Users/xies/Box/Mouse/Skin/Mesa et al/Pickled/cell_summary.pkl','rb') as file:
    wt_cells = pkl.load(file, encoding='latin1')

with open('/Users/xies/Box/Mouse/Skin/Mesa et al/Pickled/time_series.pkl','rb') as file:
    wt_ts = pkl.load(file, encoding='latin1')

#%% Collate G1 annotations

g1_annotation = pd.read_excel(path.join(dirname,'manual_track','g1_annotation.xlsx'))
g1_annotation['G1 exit'] = g1_annotation['G1 exit frame'].astype(float)

for track in tracks:
    
    corrID = track.iloc[0]['CorrID']
    if ~np.isnan(corrID):
    
        # Find same corrID in annotation
        track['G1 exit'] = g1_annotation[g1_annotation['CorrID'] == corrID]['G1 exit'].values[0]
    

ts = pd.concat(tracks)

#%% Start cell-centric dataframe with cell-level data:

df = pd.DataFrame()

for track in tracks:
    
    corrID = track.iloc[0]['CorrID']
    
    if ~np.isnan(corrID):
    
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

with open(path.join(dirname,

#%% Plot cell cycle times

plt.hist(df['Cycle length'])
plt.vlines(df['Cycle length'].mean(),ymin=0,ymax=5,color='r')
plt.xlabel('Cell cycle duration (h)')

plt.figure()
plt.hist(df['G1 length'])
plt.xlabel('G1 duration (h)')

#%%  Plot size control (time)

plt.figure()
sb.regplot(data = df, x='Birth size', y = 'Cycle length', y_jitter=True)
plt.figure()
sb.scatterplot(data = df, x='Birth size', y = 'G1 length', y_jitter=True) 
    
#%%  Plot size control (time)

def nonans(x,y):
    I = ~np.isnan(x)
    I = I & ~np.isnan(y)
    return x[I],y[I]

def pearson_r(x,y):
    x,y = nonans(x,y)
    return np.corrcoef(x,y)[0,1]

def slope(x,y):
    x,y = nonans(x,y)
    p = np.polyfit(x,y,1)
    return p[0]
    
# G1 exit
plt.figure()
sb.regplot(data = df, x='Birth size', y = 'G1 growth')
plt.xlim([-100,500])
plt.ylim([-100,500])

R = pearson_r(df['Birth size'],df['G1 growth'])
print(f'G1 Pearson R = {R}')

p = slope(df['Birth size'],df['G1 growth'])
print(f'G1 Slope m = {p}')

# Whole cycle
plt.figure()
sb.regplot(data = df, x='Birth size', y = 'Total growth')
plt.xlim([-100,500])
plt.ylim([-100,500])

R = pearson_r(df['Birth size'],df['Total growth'])
print(f'Pearson R = {R}')

p = slope(df['Birth size'],df['Total growth'])
print(f'Slope m = {p}')

    