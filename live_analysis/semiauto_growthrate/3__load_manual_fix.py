#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 22:37:33 2022

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy import stats

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

manual_track = io.imread(path.join(dirname,'manual_fix/manual_track_2.tif'))
trackIDs = np.unique(manual_track)

filenames = glob(path.join(dirname,'reg/*.tif'))
img_stack = np.array(list(map(io.imread,filenames)))

#%% Re-construct track dataframes from manual_tracks (tracking trajectories may have changed)

T,Z,Y,X = manual_track.shape

tracks = []
for trackID in tqdm(trackIDs):
    
    if trackID > 0:
        
        # Frames of interest
        mask = manual_track == trackID
        framesOI = np.where(np.any(np.any(np.any(mask,axis=1),axis=1),axis=1))[0]
        num_frames_on = len(framesOI)
        
        track = pd.DataFrame()
        
        for i,frame in enumerate(framesOI):
            
            this_mask = mask[frame,...]
            Z,Y,X = np.where(this_mask)
            
            v = this_mask.sum()
            if v == 1000: # This is the placeholder
                v = np.nan
                k10 = np.nan
            else:
                if v > 0:
                    k10 = img_stack[frame,this_mask,1].sum()
                else:
                    k10 = np.nan
            
            s = pd.Series(name = i, data = {'UniqueID': trackID
                                     ,'X': X.mean()
                                     ,'Y': Y.mean()
                                     ,'Z': Z.mean()
                                     ,'Frame': frame
                                     ,'Nuclear volume': v
                                     ,'K10 intensity': k10
                                     })
            
            #@todo: update from append, but i kind of like it like this...
            track = track.append(s)
            
        tracks.append(track)
        
with open(path.join(dirname,'manual_fix/manual_fix_tracked.pkl'),'wb') as f:
    pkl.dump(tracks, f)
    

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
    
    track['Time'] = track['Frame'] - track.iloc[0]['Frame']
    V_sm = get_interpolated_curve(track)
    
    track['Nuclear volume (sm)'] = V_sm
    track['Growth rate'] = np.hstack((np.diff(track['Nuclear volume']),np.nan))
    track['Growth rate (sm)'] = np.hstack((np.diff(track['Nuclear volume (sm)']),np.nan))
    track['Specific growth rate (sm)'] = track['Growth rate (sm)'] / track['Nuclear volume (sm)']


# Save dataframe
with open(path.join(dirname,'manual_fix/manual_fix_tracked.pkl'),'wb') as file:
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

# If any time point has K10+, then all track should be K10+
# Back propagate

for track in tracks:
    uniqueID = track.iloc[0]['UniqueID']
    track['Cell type'] = 'K10 neg'
    if np.any( ts[ts['UniqueID'] == uniqueID]['Cell type'] == 'K10 pos'):
        track['Cell type'] = 'K10 pos'

ts = pd.concat(tracks)

#%%

plt.figure()
for track in tracks:
    if track.iloc[0]['Cell type'] == 'K10 neg':
        plt.plot(track['Time'],track['Nuclear volume'],'b')
    else:
        plt.plot(track['Time'],track['Nuclear volume'],'r')

#%%

def nonans(x):
    return x[~np.isnan(x)]


sb.catplot(data= ts,x='Cell type',y='Nuclear volume',kind='violin')
sb.catplot(data= ts,x='Cell type',y='Specific growth rate (sm)',kind='violin')

print(ts.groupby('Cell type')['Nuclear volume'].mean())
print(ts.groupby('Cell type').count())

k10_pos = ts[ts['Cell type'] == 'K10 pos']
k10_neg = ts[ts['Cell type'] == 'K10 neg']

print( stats.ttest_ind(nonans(k10_pos['Specific growth rate (sm)']), nonans(k10_neg['Specific growth rate (sm)'])) )

print( ts.groupby('Cell type')['Growth rate (sm)'].mean() )
print( ts.groupby('Cell type')['Specific growth rate (sm)'].mean())

#%% Save K10 annotations for viewing

k10_annotation = np.zeros_like(manual_track)
for track in tracks:
    
    mask = manual_track == track.iloc[0]['UniqueID']
    if track.iloc[0]['Cell type'] == 'K10 pos':
        k10_annotation[mask] = 2
    elif track.iloc[0]['Cell type'] == 'K10 neg':
        k10_annotation[mask] = 1
        
io.imsave(path.join(dirname,'manual_fix/k10_annotation.tif'),k10_annotation.astype(np.int8))


    
    

