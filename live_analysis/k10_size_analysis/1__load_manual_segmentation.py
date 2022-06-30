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
dirnames['/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20200925_F7_right ear_for Shicong/'] = '*'
dirnames['/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3'] = 'cell_membrane_seg'
dirnames['/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20210322_K10 revisits/20220322_male3/area1'] = 'cell_membrane_seg'

k10_channel = {'/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20200925_F7_right ear_for Shicong/':1
               ,'/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20210322_K10 revisits/20220322_female4/area3':0
               ,'/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/20210322_K10 revisits/20220322_male3/area1':0
               }

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

        # 1. Load all the t[*].csv files for geometry information
        subname = path.split(f)[1]
        if subname[-1] == '-' or subname[-1] == '+':
            cellID = subname[0:-1]
            type_annotation = subname[-1]
        else:
            cellID = subname
            type_annotation = path.split(path.split(f)[0])[1]
            
        # Parse lineage annotation
        lineageID = subname.split('.')[0]
        
        time_points = sorted(glob(path.join(f,'t*.csv')))
        if len(time_points) == 0:
            continue
        
        # Go through each single t[X]
        track = pd.DataFrame()
        for i, this_time in enumerate(time_points):
            
            frame_str = match(r't([0-9]+)', path.basename(this_time) )[1]
            frame = float(frame_str)
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
            
            # 2. Check that 'chanX' files exist and load the intensity information
            channel_msmts = glob(path.splitext(this_time)[0] + '_chan*.txt')
            if len(channel_msmts) > 0:
                chan = k10_channel[dirname]
                chan_name = path.splitext(this_time)[0] + f'_chan{chan}.txt'
                k10_df = pd.read_csv(chan_name,sep='\t')
                k10_tot = k10_df['RawIntDen'].sum()
                k10_mean = k10_tot / df['Area'].sum()
            else:
                k10_tot = np.nan
                k10_mean = np.nan
            
            s = pd.Series(name = i,
                          data = {'CellID': cellID,'X':x,'Y':y,'Frame':frame,'Volume':V
                                  ,'Cell type': celltype ,'Dataset':dirname,'UniqueID':dirname + '/' +cellID
                                  ,'Division':division,'Birth':birth,'Leaving':leaving,'State':state
                                  ,'LineageID':lineageID,'DaughterID':'NA','ParentID':cellID,'SisterID':'NA'
                                  ,'K10 total':k10_tot,'K10 mean':k10_mean
                                  })
            
            track = track.append(s)
            
        track['Daughter'] = 'NA'
        # Load daughter cells if exist
        if path.exists(path.join(f,'a.csv')):
            daughter_a = pd.read_csv(path.join(f,'a.csv'))
            V = daughter_a['Area'].sum()
            if V > 2000:
                V = V * 0.2700001**2
            s = pd.Series(name = i+1,
                          data = {'CellID': cellID,'X':daughter_a['BX'].mean(),'Y':daughter_a['BY'].mean()
                                  ,'Frame':frame + 1,'Volume': V,'Cell type': celltype,'Dataset':dirname
                                  ,'UniqueID':dirname + '/' +cellID,'Division':False,'Birth':True,'Leaving':False
                                  ,'State':'Born','Daughter':'a','LineageID':lineageID,'DaughterID':'NA'
                                  ,'ParentID':cellID,'SisterID':cellID+'b'
                                  ,'K10 total':np.nan,'K10 mean':np.nan})
            track = track.append(s)
            
        if path.exists(path.join(f,'b.csv')):
            daughter_b = pd.read_csv(path.join(f,'b.csv'))
            V = daughter_b['Area'].sum()
            if V > 2000:
                V = V * 0.2700001**2
            s = pd.Series(name = i+2,
                          data = {'CellID': cellID,'X':daughter_a['BX'].mean(),'Y':daughter_a['BY'].mean()
                                  ,'Frame':frame + 1,'Volume': V,'Cell type': celltype,'Dataset':dirname
                                  ,'UniqueID':dirname + '/' +cellID,'Division':False,'Birth':True,'Leaving':False
                                  ,'State':'Born','Daughter':'b','LineageID':lineageID,'DaughterID':'NA'
                                  ,'ParentID':cellID,'SisterID':cellID+'a'
                                  ,'K10 total':np.nan,'K10 mean':np.nan})
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

def return_index_of_track_in_list(tracks, trackOI):
    
    uIDs = np.array([track.iloc[0]['UniqueID'] for track in tracks])
    
    I = (uIDs == trackOI.iloc[0]['UniqueID'])
    
    return np.where(I)[0][0]
    
# Annotate all the sister_pairs and also construct sister-pair list
# @todo: link the parent/child

sisters =[]
for dirname,sub_ in dirnames.items():
    
    tracks_this_area = [track for track in tracks if track.iloc[0]['Dataset'] == dirname]
    
    lineages = np.unique([track.iloc[0]['LineageID'] for track in tracks_this_area])
    
    for lin in lineages:
        
        this_lineage = [track for track in tracks_this_area if track.iloc[0]['LineageID'] == lin]
        
        # Only care if there are multiple members of lineage
        if len(this_lineage) > 0:
            
            birth_frames = np.array([track.iloc[0]['Frame'] for track in this_lineage])
            same_frame,num_times_shared = np.unique(birth_frames, return_counts=True)
            
            if np.any(num_times_shared == 2):
                frame_shared = same_frame[np.where(num_times_shared == 2)]
                # Cells born on this frame are sisters
                I = np.where(birth_frames == frame_shared)[0]
                #@todo: figure out how avoid 'look up' but for now it should work
                sister_a = this_lineage[I[0]]
                sister_b = this_lineage[I[1]]
                tracks[return_index_of_track_in_list(tracks,sister_a)].at[:,'SisterID'] = sister_b['CellID']
                tracks[return_index_of_track_in_list(tracks,sister_b)].at[:,'SisterID'] = sister_a['CellID']
                
                sisters.append([sister_a, sister_b])

# Put the a/b annotated sisters into list as well
#@todo: need to integrate
for track in tracks:
    # Both daughter are present
    if np.any(track['Daughter'] == 'a') and np.any(track['Daughter'] == 'b'):
        sisters.append([track[track['Daughter'] == 'a'], track[track['Daughter'] == 'b']])


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
import pickle as pkl
# Save dataframe
with open('/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/tracks.pkl','wb') as file:
    pkl.dump(tracks,file)

