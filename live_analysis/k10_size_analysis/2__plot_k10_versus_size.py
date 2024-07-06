#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:37:34 2022

@author: xies
"""

import pandas as pd
import seaborn as sb
import pickle as pkl

with open('/Users/xies/OneDrive - Stanford/Skin/Two photon/Shared/tracks.pkl', 'rb') as f:
    tracks = pkl.load(f)


#%% Raw growth curves

plt.figure()
for track in tracks:
    track_ = track[track['Daughter'] == 'NA']
    if track_.iloc[0]['Cell type'] == 'K10 neg':
        plt.plot(track_['Time'],track_['Volume'],'b')
    else:
        plt.plot(track_['Time'],track_['Volume'],'r')
        
plt.xlabel('Time (not aligned by birth) (days)')
plt.ylabel('Cell volume (um3)')
plt.legend(['K10 neg', 'K10 pos'])

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

sb.catplot(data = ts,x='State',y='Volume',kind='strip',hue='Cell type',split=True)

# sb.catplot(data= ts,x='Cell type',y='Specific growth rate (sm)',hue='Dataset',
#            kind='strip', split=True)


# print(ts.groupby('Cell type')['Volume'].mean())
print(ts.groupby('Cell type')['Specific growth rate (sm)'].mean())
print(birth.groupby('Cell type').count())
print(division.groupby('Cell type').count())

print('-----Volume------')
# print( stats.ttest_ind(nonans(k10_pos['Volume']), nonans(k10_neg['Volume'])) )
print(groupby_ttest(ts,'Cell type','Volume'))

print(groupby_ttest(birth,'Cell type','Volume'))
print(groupby_ttest(na,'Cell type','Volume'))
print(groupby_ttest(division,'Cell type','Volume'))

print('-----Specific growth rate------')
print(groupby_ttest(ts,'Cell type','Specific growth rate (sm)'))


#%%

plt.figure()

# plt.scatter(ts['Volume'],ts['Specific growth rate (sm)'], alpha = 0.1)

sb.lmplot(data= ts, x='Volume', y='Specific growth rate (sm)',hue='Cell type')
plt.xlim([200,1000])

#%% Sister analysis

# How often are they the same cell type?

fate_diff = np.zeros(len(sisters))
size_diff = np.zeros(len(sisters))
total_size = np.zeros(len(sisters))
for i,pair in enumerate(sisters):
    
    a = pair[0]
    b = pair[1]
    
    if a.iloc[0]['Cell type'] == b.iloc[0]['Cell type']:
        fate_diff[i] = 0
    else:
        fate_diff[i] = 1
        
    size_diff[i] = abs(a.iloc[0]['Volume'] - b.iloc[0]['Volume'])
    total_size[i] = a.iloc[0]['Volume'] + b.iloc[0]['Volume']
        
#%%

def standardize(X):
    return (X-X.mean())/X.std()

ts = pd.concat(tracks,ignore_index=True)

for dataset in np.unique( ts['Dataset']):
    ts.at[ts['Dataset'] == dataset, 'K10 norm'] = standardize(ts[ts['Dataset'] == dataset]['K10 mean'])
    
birth = ts[ts['Birth'] == 1]
sb.lmplot(data=birth,x='Volume',y='K10 norm',hue='Cell type', fit_reg=False)
plt.xlabel('Volume at birth')
plt.ylabel('K10 intensity (mean)')


