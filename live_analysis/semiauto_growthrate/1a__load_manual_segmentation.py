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
import seaborn as sb
from scipy import stats

dirnames = {}
# dirnames['/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area1'] = 'cell_membrane_seg'
dirnames['/Users/xies/Box/Mouse/Skin/Two photon/Shared/20200925_F7_right ear_for Shicong/'] = '*'
dirnames['/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3'] = 'cell_membrane_seg'
dirnames['/Users/xies/Box/Mouse/Skin/Two photon/Shared/20210322_K10 revisits/20220322_male3/area1'] = 'cell_membrane_seg'

#%% Parse .csv files
'''

../CellID[+]/t{frame}.csv
../CellID[-]/t{frame}.csv
../CellID[-]/t{frame}.b.csv -- birth
../CellID[-]/t{frame}.d.csv -- division
../CellID[-]/t{frame}.l.csv -- leaving

'''

tracks = []
    
for dirname,sub_ in dirnames.items():
    
    print(f'{dirname}')
    all_cells = glob(path.join(dirname,sub_,'*'))
    
    for f in all_cells:
        
        # skip .tif files
        if path.splitext(f)[1] == '.tif':
            continue
        
        subname = path.split(f)[1]
        if subname[-1] == '-' or subname[-1] == '+':
            cellID = subname[0:-1]
            type_annotation = subname[-1]
        else:
            cellID = subname
            type_annotation = path.split(path.split(f)[0])[1]
            
        # Parse lineage annotation
        lineageID = subname.split('.')[0]
        
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
            basename = path.basename(this_time)
            if len(basename.split('.')) == 3:
                if basename.split('.')[-2] == 'd':
                    division = True
                    birth = False
                    leaving = False
                    state = 'Division'
                elif basename.split('.')[-2] == 'b':
                    division = False
                    birth = True
                    leaving = False
                    state = 'Born'
                elif basename.split('.')[-2] == 'l':
                    division = False
                    birth = False
                    leaving = True
                    state = 'Delaminating'
            else:
                birth = False
                division = False
                leaving = False
            
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
                                  ,'LineageID':lineageID
                                  })
            
            track = track.append(s)
            
        track['Daughter'] = 'NA'
        # Load daughter cells if exists
        if path.exists(path.join(f,'a.csv')):
            daughter_a = pd.read_csv(path.join(f,'a.csv'))
            V = daughter_a['Area'].sum()
            if V > 2000:
                V = V * 0.2700001**2
            s = pd.Series(name = i+1,
                          data = {'CellID': cellID
                                  ,'X':daughter_a['BX'].mean()
                                  ,'Y':daughter_a['BY'].mean()
                                  ,'Frame':frame + 1
                                  ,'Volume': V
                                  ,'Cell type': celltype
                                  ,'Dataset':dirname
                                  ,'Division':False
                                  ,'Birth':True
                                  ,'Leaving':False
                                  ,'State':'Born'
                                  ,'Daughter':'a'
                                  ,'LineageID':lineageID
                                  })
            track = track.append(s)
            
        if path.exists(path.join(f,'b.csv')):
            daughter_b = pd.read_csv(path.join(f,'b.csv'))
            V = daughter_b['Area'].sum()
            if V > 2000:
                V = V * 0.2700001**2
            s = pd.Series(name = i+2,
                          data = {'CellID': cellID
                                  ,'X':daughter_a['BX'].mean()
                                  ,'Y':daughter_a['BY'].mean()
                                  ,'Frame':frame + 2
                                  ,'Volume': V
                                  ,'Cell type': celltype
                                  ,'Dataset':dirname
                                  ,'Division':False
                                  ,'Birth':True
                                  ,'Leaving':False
                                  ,'State':'Born'
                                  ,'Daughter':'b'
                                  ,'LineageID':lineageID
                                  })
            track = track.append(s)
            
        track['Divides'] = np.any(track['Division'])
        track['Born'] = np.any(track['Birth'])
        track['Leaves'] = np.any(track['Leaving'])
        tracks.append(track)


tracks_div = [track for track in tracks if track.iloc[0]['Divides']]
tracks_non_div = [track for track in tracks if not track.iloc[0]['Divides']]
tracks_leaves = [track for track in tracks if track.iloc[0]['Leaves']]
tracks_not_leaving = [track for track in tracks if not track.iloc[0]['Leaves']]

#%% Lineage reconstruct: 1) annotate mother/daughter 2) annotate sister


#%% Growth rate calculation / spline smooth

from scipy.interpolate import UnivariateSpline
def get_interpolated_curve(track,smoothing_factor=1e5):

    Idaughter= track['Daughter'] == 'NA'
    track_ = track[Idaughter]
    v = track_['Volume']
    if (~np.isnan(v)).sum() < 3:
        yhat = v
        
    else:
        t = track_['Frame'] * 24
        # Spline smooth
        spl = UnivariateSpline(t, v, k=2, s=smoothing_factor)
        yhat = spl(t)

    # Pad NAN for daughter cells
    num_daughters = (~Idaughter).sum()
    yhat = np.hstack((yhat,np.ones(num_daughters) * np.nan))
    
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
    track_ = track[track['Daughter'] == 'NA']
    if track_.iloc[0]['Cell type'] == 'K10 neg':
        plt.plot(track_['Time'],track_['Volume'],'b')
    else:
        plt.plot(track_['Time'],track_['Volume'],'r')

#%%

# tracks = tracks_div

def groupby_ttest(df,groupname,y_value):
    df_ = df[~np.isnan(df[y_value])]
    return stats.ttest_ind(*df_.groupby(groupname)[y_value].apply(lambda x:list(x)))

def nonans(x):
    return x[~np.isnan(x)]

ts = pd.concat(tracks_not_leaving)
division = ts[ts['Division'] == 1]
birth = ts[ts['Birth'] == 1]
na = ts[ts['State'] == 'NA']
not_na = ts[ts['State'] != 'NA']

sb.catplot(data = na,hue='Cell type',y='Volume',kind='strip',split=True,x='Dataset')

sb.catplot(data = division,hue='Cell type',y='Volume',kind='strip',split=True,x='Dataset')
sb.catplot(data = birth,hue='Cell type',y='Volume',kind='strip',split=True,x='Dataset')

# sb.catplot(data= ts,x='Cell type',y='Specific growth rate (sm)',hue='Dataset',
#            kind='strip', split=True)


# print(ts.groupby('Cell type')['Volume'].mean())
print(ts.groupby('Cell type')['Specific growth rate (sm)'].mean())
print(birth.groupby('Cell type').count())
print(division.groupby('Cell type').count())

print('-----Volume------')
# print( stats.ttest_ind(nonans(k10_pos['Volume']), nonans(k10_neg['Volume'])) )
print(groupby_ttest(birth,'Cell type','Volume'))
print(groupby_ttest(division,'Cell type','Volume'))

print('-----Specific growth rate------')
print(groupby_ttest(ts,'Cell type','Specific growth rate (sm)'))


#%%

plt.figure()

# plt.scatter(ts['Volume'],ts['Specific growth rate (sm)'], alpha = 0.1)

sb.lmplot(data= ts, x='Volume', y='Specific growth rate (sm)',hue='Cell type')




