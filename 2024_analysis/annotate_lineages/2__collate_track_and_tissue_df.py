#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:54:57 2025

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from skimage import io
from glob import glob
from natsort import natsort

from mathUtils import fit_exponential_curve
import seaborn as sb
from os import path

import pickle as pkl

dx = 0.25
dz = 1

#%% Load tracks

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Skin/Mesa et al/W-R1/'

with open(path.join(dirname,'Mastodon/dense_tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)

#%% Load cyto segmentations
# Go through each lineage and write TrackID -> CellposeID

# NB: Note where there is no segmentation
filenames = natsort.natsorted(glob(path.join(dirname,'3d_cyto_seg/3d_cyto_manual/t*.tif')))
cyto_segs = np.stack( list(map(io.imread, filenames) ) )

for trackID,track in enumerate(tracks):
    
    track['Frame'] = track['Frame'].astype(float)
    track['Movie time'] = track['Frame'] * 12
    
    if not np.isnan( track.iloc[0]['Mother'] ):
        track['Age'] = track['Frame'] - track.iloc[0]['Frame']
        
    if track.iloc[0]['Complete cycle']:
        track['Time to division'] = np.array(range(-len(track)+1,1))*12    
        
    if track.iloc[0]['Differentiated']:
        track['Time to delamination'] = np.array(range(-len(track)+1,1))*12

    for idx,spot in track.iterrows():
        
        X = int( float(spot['X']) / dx)
        Y = int( float(spot['Y']) / dx)
        Z = int( float(spot['Z']) / dz)
        t = int( float(spot['Frame']) )
        
        cytoID = cyto_segs[t,Z,Y,X]
        track.loc[idx,'CellposeID'] = cytoID
        
df_tracks = pd.concat(tracks,ignore_index=True)
df_tracks['Frame'] = df_tracks['Frame'].astype(float)
df_tracks = df_tracks.rename(columns={'ID':'SpotID'})

# Merge on CellposeID
df = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)
fields2drop = ['X','Y','Z','Left','Right','Division','Terminus']
df = df.merge(df_tracks.drop(columns=fields2drop),on = ['CellposeID','Frame'])

# Merge in the cell cycle phase
basals = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
df = df.merge(basals[['CellposeID','Frame','Phase','Time to G1S']],
         on=['CellposeID','Frame'], how='outer')

df = df.sort_values(['TrackID','Frame'])

df = df[~(df.Cutoff).astype(bool)]

#%%

tracks = []
for trackID,track in df.groupby('TrackID'):
    if len(track.dropna(subset='Cell volume')) > 2:
        t = track['Movie time'].values
        v = track['Cell volume'].values
        try:
            gamma = fit_exponential_curve(t,v)[0][1]
        except RuntimeError:
            gamma = np.nan
    else:
        gamma = np.nan
    track['Exponential growth rate'] = gamma/12
    tracks.append(track)
df = pd.concat(tracks,ignore_index=True)

#%%

def get_track_with_spotID(df,spotID):
    trackID = df[df['SpotID'] == spotID]['TrackID']
    track = df[df['TrackID'] == trackID]
    return track

def get_mother_track(df,daughter):
    motherID = daughter.iloc[0]['Mother']
    mother = df[df['TrackID'] == motherID]
    return mother

def get_sister_track(df,sister):
    motherID = sister.iloc[0]['Sister']
    mother = df[df['TrackID'] == motherID]
    return mother

plt.figure(1)
for _,track in df.groupby('TrackID'):
    
    mother = get_mother_track(df,track)
    if len(mother) > 0:
        if track.iloc[0]['Differentiated']:
            plt.subplot(1,2,1)
            plt.plot(mother.Age,mother['Cell volume'],'b',alpha=0.1)
            plt.ylim([200,900])
        elif track.iloc[0]['Complete cycle']:
            plt.subplot(1,2,2)
            plt.plot(mother.Age,mother['Cell volume'],'r',alpha=0.1)
            plt.ylim([200,900])
plt.xlabel('Cell age (h)')

#%%

cells = pd.DataFrame()
cells['TrackID'] = [trackID for trackID,t in df.groupby('TrackID')]
cells['Complete cycle'] = [t.iloc[0]['Complete cycle'] for _,t in df.groupby('TrackID')]
cells['Differentiated'] = [t.iloc[0]['Differentiated'] for _,t in df.groupby('TrackID')]

cells['Birth volume'] = [
    t.iloc[0]['Cell volume'] if ~np.isnan(t.iloc[0]['Mother']) else np.nan
    for _,t in df.groupby('TrackID')]
cells['Birth apical area'] = [ t.iloc[0]['Apical area'] for _,t in df.groupby('TrackID')]
cells['Birth basal area'] = [ t.iloc[0]['Basal area'] for _,t in df.groupby('TrackID')]
cells['Birth tissue curvature'] = [ t.iloc[0]['Mean curvature'] for _,t in df.groupby('TrackID')]

cells['Mother growth rate'] = [
    get_mother_track(df,t).iloc[-1]['Exponential growth rate']
    if len(get_mother_track(df,t)) > 0
    else np.nan
    for _,t in df.groupby('TrackID')]
cells['Mother division volume'] = [
    get_mother_track(df,t).iloc[-1]['Cell volume']
    if len(get_mother_track(df,t)) > 0
    else np.nan
    for _,t in df.groupby('TrackID')]
cells['Mother division curvature'] = [
    get_mother_track(df,t).iloc[-1]['Mean curvature']
    if len(get_mother_track(df,t)) > 0
    else np.nan
    for _,t in df.groupby('TrackID')]

cells['Sister birth size difference'] = [
    np.abs(get_sister_track(df,t).iloc[-1]['Cell volume'] - df.iloc[0]['Cell volume'])
    if len(get_sister_track(df,t)) > 0
    else np.nan
    for _,t in df.groupby('TrackID')]



sb.catplot(cells,x='Differentiated',y='Birth volume',kind='box')
sb.catplot(cells,x='Differentiated',y='Birth tissue curvature',kind='box')
sb.catplot(cells,x='Differentiated',y='Mother division curvature',kind='box')
sb.catplot(cells,x='Differentiated',y='Mother growth rate',kind='box')
sb.catplot(cells,x='Differentiated',y='Sister birth size difference',kind='box')




