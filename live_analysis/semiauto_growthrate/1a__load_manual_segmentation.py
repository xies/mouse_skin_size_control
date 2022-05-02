#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 22:28:50 2022

@author: xies
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from skimage import io
from os import path
from glob import glob
import pandas as pd
from re import match

dirnames = {
            '/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1':'cell_membrane_seg'
           ,'/Users/xies/Box/Mouse/Skin/Two photon/Shared/20200925_F7_right ear_for Shicong/':'*'
           ,'/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3':'cell_membrane_seg'}

#%%

tracks = []
    
for dirname,sub_ in dirnames.items():
    
    print(f'{dirname}')
    all_cells = glob(path.join(dirname,sub_,'*'))
    
    for f in all_cells:
        
        subname = path.split(f)[1]
        if subname[-1] == '-' or subname[-1] == '+':
            cellID = subname[0:-1]
            type_annotation = subname[-1]
        else:
            cellID = subname
            type_annotation = path.split(path.split(f)[0])[1]
            
        time_points = glob(path.join(f,'t*.csv'))
        if len(time_points) == 0:
            continue
        
        track = pd.DataFrame()
        for i, this_time in enumerate(time_points):
            
            frame = float(match(r't([0-9]+)', path.basename(this_time) )[1])
            # Parse cell type
            if type_annotation == '+':
                celltype = 'K10 pos'
            elif type_annotation == '-':
                celltype = 'K10 neg'
            elif type_annotation == 'K10neg_divisions':
                celltype = 'K10 neg'
            elif type_annotation == 'K10pos_divisions':
                celltype = 'K10 pos'
            else:
                break
            # Parse cell cycle annotation (if applicable)
            state = 'NA'
            if len(this_time.split('.')) == 3:
                if this_time.split('.')[1] == 'd':
                    division = True
                    birth = False
                    leaving = False
                    state = 'Division'
                elif this_time.split('.')[1] == 'b':
                    division = False
                    birth = True
                    leaving = False
                    state = 'Born'
                elif this_time.split('.')[1] == 'l':
                    division = False
                    birth = False
                    leaving = True
                    state = 'Delaminating'
            else:
                birth = False
                division = False
            
            df = pd.read_csv(this_time)
            V = df['Area'].sum()
            if V > 2000:
                V = V * 0.2700001**2
            x = df['BX'].mean()
            y = df['BY'].mean()
            s = pd.Series(name = i,
                          data = {'CellID': cellID
                                  ,'X':x
                                  ,'Y':y
                                  ,'Frame':frame
                                  ,'Volume':V
                                  ,'Cell type': celltype 
                                  ,'Dataset':dirname
                                  ,'Division':division
                                  ,'Birth':birth
                                  ,'Leaving':leaving
                                  ,'State':state
                                  })
            
            track = track.append(s)
        
        track['Divides'] = np.any(track['Division'])
        track['Born'] = np.any(track['Birth'])
        track['Leaves'] = np.any(track['Leaving'])
        tracks.append(track)


tracks_div = [track for track in tracks if track.iloc[0]['Divides']]
tracks_non_div = [track for track in tracks if not track.iloc[0]['Divides']]
tracks_leaves = [track for track in tracks if track.iloc[0]['Leaves']]

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
    track['Growth rate (sm)'] = np.hstack((np.diff(track['Volume (sm)']),np.nan)) / np.hstack((np.diff(track['Frame']),np.nan)) / 24
    track['Specific growth rate (sm)'] = track['Growth rate (sm)'] / track['Volume (sm)']

# tracks = tracks_div
# tracks = tracks_non_div

# Save dataframe
# with open(path.join(dirname,'cell_membrane_seg/tracks.pkl'),'wb') as file:
#     pkl.dump(tracks,file)

#%%

plt.figure()
for track in tracks:
    if track.iloc[0]['Cell type'] == 'K10 neg':
        plt.plot(track['Time'],track['Volume'],'b')
    else:
        plt.plot(track['Time'],track['Volume'],'r')

#%%

# tracks = tracks_div

def nonans(x):
    return x[~np.isnan(x)]

ts = pd.concat(tracks)
division = ts[ts['Division'] == 1]
birth = ts[ts['Birth'] == 1]

sb.catplot(data= ts,x='Cell type',y='Volume',kind='strip',split=True,hue='State')
# sb.catplot(data= ts,x='Cell type',y='Volume',kind='violin',split=True)
sb.catplot(data= ts,x='Cell type',y='Specific growth rate (sm)',
           kind='strip', split=True)


print(ts.groupby('Cell type')['Volume'].mean())
print(ts.groupby('Cell type')['Specific growth rate (sm)'].mean())
print(ts.groupby('Cell type').count())

k10_pos = ts[ts['Cell type'] == 'K10 pos']
k10_neg = ts[ts['Cell type'] == 'K10 neg']

print('-----Volume------')
print( stats.ttest_ind(nonans(k10_pos['Volume']), nonans(k10_neg['Volume'])) )


print('-----Specific growth rate------')
print( stats.ttest_ind(nonans(k10_pos['Specific growth rate (sm)']), nonans(k10_neg['Specific growth rate (sm)'])) )

# print( ts.groupby('Cell type')['Growth rate (sm)'].mean() )
# print( ts.groupby('Cell type')['Specific growth rate (sm)'].mean())

#%%

plt.figure()

# plt.scatter(ts['Volume'],ts['Specific growth rate (sm)'], alpha = 0.1)

sb.lmplot(data= ts, x='Volume', y='Specific growth rate (sm)',hue='Cell type')




