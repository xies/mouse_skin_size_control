#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:02:33 2024

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import matplotlib.pyplot as plt
from tqdm import tqdm
from mathUtils import parse_3D_inertial_tensor, surface_area

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'

dx = 0.26
dz = 2

#%%

filt_seg = io.imread(path.join(dirname,'manual_cellcycle_annotations/filtered_segs.tif'))

T = filt_seg.shape[0]
df = []
for t in tqdm(range(T)):
    props = measure.regionprops(filt_seg[t,...])
    _df = pd.DataFrame(index=range(len(props)),columns=['trackID', 'Nuclear volume'
                                                        ,'Axial moment','Axial angle'
                                                        ,'Planar moment 1'
                                                        ,'Planar moment 2'
                                                        ,'Planar angle'
                                                        ,'Z','Y','X'])
    for i,p in enumerate(props):
        _df.loc[i,'trackID'] = p['label']
        _df.loc[i,'Nuclear volume'] = p['area'] * dx**2 * dz
        I = p['inertia_tensor']
        Iaxial, phi, Ia, Ib, theta = parse_3D_inertial_tensor(I)
        z,y,x = p['centroid']
        _df.loc[i,'Z'] = z
        _df.loc[i,'Y'] = y
        _df.loc[i,'X'] = x
        
        _df['Frame'] = t
        
    df.append(_df)
    
df = pd.concat(df,ignore_index=True)
df = df.sort_values(['trackID','Frame'])
df['Time'] = df['Frame'] * 10
df = df.reset_index()

#%%

tracking_df = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/filtered_tracks.csv'),index_col=0)
annos = {trackID:cell for trackID,cell in tracking_df.groupby('TrackID')}
tracks = {trackID:cell for trackID,cell in df.groupby('trackID')}

for trackID,t in tracks.items():
    
    # Truncate since didn't do whole time course yet
    this_anno = annos[trackID]
    this_anno = this_anno.rename(columns={'FRAME':'Frame'})
    t = pd.merge(t,this_anno[['Phase','Frame']],on='Frame')
    tracks[trackID] = t

df = pd.concat(tracks,ignore_index=True)

for t in tracks.values():
    plt.plot(t.Frame,t['Nuclear volume'])

#%%

size_summary = df.groupby(['trackID','Phase']).mean()['Nuclear volume']
size_summary = size_summary.reset_index()

size_summary = pd.pivot(size_summary,index='trackID',columns=['Phase'])
size_summary.columns = size_summary.columns.droplevel(0)
size_summary['G1 growth'] = size_summary['G1S'] - size_summary['Visible birth']

plt.scatter(size_summary['Visible birth'],size_summary['G1 growth'])

    
    
    
    