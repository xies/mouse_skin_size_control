#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:28:50 2022

@author: xies
"""

import numpy as np
from skimage import io
from os import path
from glob import glob
import pandas as pd
from re import match

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1'
dirname = '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20200925_F7_right ear_for Shicong/K10neg_divisions'


#%%

all_cells = glob(path.join(dirname,'cell_membrane_seg/*/'))

tracks = []

for f in all_cells:
    
    subname = path.split( path.split(f)[0] )[1]
    cellID = subname[0:-1]
    type_annotation = subname[-1]
    
    time_points = glob(path.join(f,'*.csv'))
    track = pd.DataFrame()
    for i, this_time in enumerate(time_points):
        
        frame = float(match(r't(\d+).csv', path.basename(this_time) )[1])
        if type_annotation == '+':
            celltype = 'K10 pos'
        elif type_annotation == '-':
            celltype = 'K10 neg'
        else:
            break
        
        df = pd.read_csv(this_time)
        V = df['Area'].sum()
        x = df['BX'].mean()
        y = df['BY'].mean()
        s = pd.Series(name = i,
                      data = {'CellID': cellID
                              ,'X':x
                              ,'Y':y
                              ,'Frame':frame
                              ,'Volume':V
                              ,'Cell type': celltype })
        track = track.append(s)
        
    tracks.append(track)
    
#%%

from scipy.interpolate import UnivariateSpline
def get_interpolated_curve(track,smoothing_factor=1e5):

    v = track['Volume']
    if (~np.isnan(v)).sum() < 3:
        yhat = v
        
    else:
        t = track['Frame'] * 24
        # Spline smooth
        spl = UnivariateSpline(t, v, k=2, s=smoothing_factor)
        yhat = spl(t)

    return yhat
    


for track in tracks:
    
    track['Time'] = track['Frame'] - track.iloc[0]['Frame']
    V_sm = get_interpolated_curve(track)
    track['Volume (sm)'] = V_sm
    track['Growth rate'] = np.hstack((np.diff(track['Volume']),np.nan))
    track['Growth rate (sm)'] = np.hstack((np.diff(track['Volume (sm)']),np.nan))
    track['Specific growth rate (sm)'] = track['Growth rate (sm)'] / track['Volume (sm)']


# Save dataframe
with open(path.join(dirname,'cell_membrane_seg/tracks.pkl'),'wb') as file:
    pkl.dump(tracks,file)

#%%

plt.figure()
for track in tracks:
    if track.iloc[0]['Cell type'] == 'K10 neg':
        plt.plot(track['Time'],track['Volume'],'b')
    else:
        plt.plot(track['Time'],track['Volume'],'r')

#%%

def nonans(x):
    return x[~np.isnan(x)]

ts = pd.concat(tracks)

sb.catplot(data= ts,x='Cell type',y='Volume',kind='violin')
sb.catplot(data= ts,x='Cell type',y='Specific growth rate (sm)',kind='strip')

print(ts.groupby('Cell type')['Volume'].mean())
print(ts.groupby('Cell type').count())

k10_pos = ts[ts['Cell type'] == 'K10 pos']
k10_neg = ts[ts['Cell type'] == 'K10 neg']

print( stats.ttest_ind(nonans(k10_pos['Specific growth rate (sm)']), nonans(k10_neg['Specific growth rate (sm)'])) )

print( ts.groupby('Cell type')['Growth rate (sm)'].mean() )
print( ts.groupby('Cell type')['Specific growth rate (sm)'].mean())


