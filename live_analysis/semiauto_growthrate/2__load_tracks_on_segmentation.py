#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 23:15:08 2022

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import seaborn as sb
from os import path
from glob import glob

import pickle as pkl
from skimage import io
from tqdm import tqdm

from roipoly import roipoly

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1'

# Load assembled tracks as pkl

with open(path.join(dirname,'MaMut/tracks.pkl'),'rb') as file:
    tracks = pkl.load(file)
 

#%% Load images

filenames = glob(path.join(dirname,'stardist/prediction/*/*.tif'))

seg = np.array(list(map(io.imread,filenames)))
seg_tracked = np.zeros_like(seg)

filenames = glob(path.join(dirname,'reg/*.tif'))

img_stack = np.array(list(map(io.imread,filenames)))

#%% Load segmented images

for track in tqdm(tracks):
    
    # Frames of interest
    framesOI = track['Frame']
    
    for i,frame in enumerate(framesOI):
        
        if frame > 6: # Missing last two frames
            continue
        
        t = int(frame)
        this_seg = seg[t,...]
        x = int(track.iloc[i]['X'])
        y = int(track.iloc[i]['Y'])
        z = int(track.iloc[i]['Z'])
        
        this_label = this_seg[z,y,x]
        if this_label > 0:
            mask = this_seg == this_label
            
            seg_tracked[t,mask] = track.iloc[0]['UniqueID']

io.imsave(path.join(dirname,'tracked_seg','track_seg_raw.tif'), seg_tracked.astype(np.int16))

#%% Extract segmented volume (and K10 intensity) from images

dx = 0.25

for track in tqdm(tracks):
    
    framesOI = track['Frame']
    num_frames_on = len(framesOI)
    
    volumes = np.ones(num_frames_on) * np.nan
    k10 = np.ones(num_frames_on) * np.nan
    for i, frame in enumerate(framesOI):
        # Missing last two frames, so filtera
        t = int(frame)
        
        if t > 6:
            continue
        
        mask = seg_tracked[t,...] == track.iloc[0]['UniqueID']
        v = mask.sum()
        if v > 0:
            volumes[i] = v #Nuc volume in pixels
            k10[i] = img_stack[t,mask,1].sum()

    # Update in place
    track['Nuclear volume'] = np.array(volumes) * dx ** 2
    track['K10 intensity'] = k10
        

#%% Calculate raw growth curves

from scipy.interpolate import UnivariateSpline
def get_interpolated_curve(track,smoothing_factor=1e5):

    v = track['Nuclear volume']
    if (~np.isnan(v)).sum() < 3:
        yhat = v
        
    else:
        t = track['Frame'] * 12
        # Spline smooth
        spl = UnivariateSpline(t, v, k=2, s=smoothing_factor)
        yhat = spl(t)

    return yhat
    


for track in tracks:
    
    V_sm = get_interpolated_curve(track)
    track['Nuclear volume (sm)'] = V_sm
    track['Growth rate'] = np.hstack((np.diff(track['Nuclear volume']),np.nan))
    track['Growth rate (sm)'] = np.hstack((np.diff(track['Nuclear volume (sm)']),np.nan))


# Save dataframe
with open(path.join(dirname,'MaMut/tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)


#%% Plot

def gate_cells(df,x,y,gate):
    from matplotlib import path as mplPath
    
    gate_p = mplPath.Path( np.vstack((gate.x,gate.y)).T )
    gateI = gate_p.contains_points( np.vstack((df[x],df[y])).T )
    return gateI

ts = pd.concat(tracks)

# K10 gating
plt.scatter(ts['Nuclear volume'],ts['K10 intensity'], alpha = 0.1)
gate_ = roipoly()
I = gate_cells(ts,'Nuclear volume','K10 intensity',gate_)

ts['Cell type'] = 'K10 neg'
ts.loc[I,'Cell type'] = 'K10 pos'


#%%

sb.catplot(data= ts,x='Cell type',y='Nuclear volume')
sb.catplot(data= ts,x='Cell type',y='Growth rate')



