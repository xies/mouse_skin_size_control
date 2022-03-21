#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:30:42 2022

@author: xies
"""

import numpy as np
import pandas as pd


with open('/Users/xies/Box/Mouse/Skin/Mesa et al/tracked_data_collated/cell_summary.pkl','rb' ) as f:
    df = pkl.load(f)

with open('/Users/xies/Box/Mouse/Skin/Mesa et al/tracked_data_collated/time_series.pkl','rb' ) as f:
    ts = pkl.load(f)
    
#%% Convert 12h sampling rate to 24h sampling rate

# Pick the even frames only

ts_daily = []
for this_ts in ts:
    ts_daily.append( this_ts[ (this_ts['Frame'] % 2) == 1] )
    
# Reconstruct the cell-centric dataframe
cells_ = pd.DataFrame(columns = ['CellID','Birth frame','G1 frame','Division frame','Birth volume','G1 volume','Division volume'])

for i,this_ts in enumerate(ts_daily):
    
    this_ts = this_ts[this_ts['Daughter'] == 'None']
    
    bframe = this_ts.iloc[0]['Frame']
    bsize = this_ts.iloc[0]['Volume']
    
    dframe = this_ts.iloc[-1]['Frame']
    dvolume = this_ts.iloc[-1]['Volume']
    
    if np.any(this_ts['Phase'].values != 'G1'):
        Ig1 = np.where(this_ts['Phase'] != 'G1')[0][0]
        g1frame = this_ts.iloc[Ig1]['Frame']
        g1volume = this_ts.iloc[Ig1]['Volume']
        
    else:
        g1frame = np.nan
        g1volume = np.nan
    
    cells_ = cells_.append( pd.Series({'CellID': this_ts.iloc[0]['CellID'],
                             'Birth frame': bframe,
                             'Birth volume': bsize,
                             'Division frame':dframe,
                             'Division volume': dvolume,
                             'G1 frame': g1frame,
                             'G1 volume': g1volume}), ignore_index = True)
    
cells_['G1 length'] = (cells_['G1 frame'] - cells_['Birth frame']) * 12
cells_['G1 growth'] = cells_['G1 volume'] - cells_['Birth volume']
cells_['Cell cycle length'] = (cells_['Division frame'] - cells_['Birth frame']) * 12
cells_['Total growth'] = cells_['Division volume'] - cells_['Birth volume']
    
cells_.at[cells_['G1 growth'] == 0,'G1 growth'] = np.nan

#%% Plot size control

m = df.mean()['Cycle length']
print(f'Mean cell cycle time (full sample): {m}')
m = df.mean()['G1 length']
print(f'Mean G1 time (full sample): {m}')


m = cells_.mean()['Cell cycle length']
print(f'Mean cell cycle time (half sample): {m}')
m = cells_.mean()['G1 length']
print(f'Mean G1 time (half sample): {m}')

plt.figure()
plt.scatter(df['Birth volume'],df['G1 grown'])
plt.scatter(cells_['Birth volume'],cells_['G1 growth'])
plt.xlabel('Birth volume')
plt.ylabel('G1 growth')
plt.legend(['12h sampling rate','24h sampling rate'])


plt.figure()
plt.scatter(df['Birth volume'],df['Total growth'])
plt.scatter(cells_['Birth volume'],cells_['Total growth'])
plt.xlabel('Birth volume')
plt.ylabel('Total growth')
plt.legend(['12h sampling rate','24h sampling rate'])


