#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:35:49 2023

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

from os import path
from glob import glob
from tqdm import tqdm
import pickle as pkl

dirnames = {}
dirnames['M1R1'] = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
dirnames['M1R2'] = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/'

df = []
ts = []
for name,dirname in dirnames.items():
    _df = pd.read_csv(path.join(dirname,'MLR model/ts_features.csv'),index_col=0)
    _df['Region'] = name
    df.append(_df)
    _ts = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col = 0)
    _df['Region'] = name
    ts.append(_ts)

df = pd.concat(df,ignore_index=True)
ts = pd.concat(ts,ignore_index=True)

#%% Illustrate the problem

plt.figure(1)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Volume (sm)'])


plt.xlabel('Movie frame')
plt.ylabel('Cell volume (fL)')

plt.figure(2)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Nuclear volume'])

plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume (fL)')


plt.figure(3)
mesa = dfc
mesa['Frame'] = mesa['Frame'] - 1
for cellID in mesa['CellID'].unique():
    this_cell = mesa[mesa['CellID'] == cellID]
    this_cell = this_cell[this_cell['Daughter'] == 'None']
    plt.plot(this_cell.Frame,this_cell['Nucleus'])

plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume -- Otsu thresholded (fL)')

#%%

df_all = df.rename(columns={'basalID':'CellID'}).merge(mesa,on=['Region','CellID','Frame'])
# df_all = df_all[df_all['Daughter'] == 'None']
sb.lmplot(data = df_all, x= 'Nucleus',y='Nuclear volume',hue='Frame',
          facet_kws = {'sharex':True, 'sharey':True})
plt.xlim([0,325])
plt.ylim([0,325])

#%%

plt.figure(4)
sb.lmplot(data=df,x='Volume (sm)', y='Nuclear volume', col='Frame', col_wrap=5,
          facet_kws=dict(sharex=True, sharey=True), robust=False)

plt.figure(5)
sb.lmplot(data=mesa,x='Volume (sm)', y='Nucleus', col='Frame', col_wrap=5,
          facet_kws=dict(sharex=True, sharey=True), robust=False)

#%%

sb.catplot(data=ts,x='Frame',y='Volume',kind = 'violin')

mean_vol = np.zeros(15)
for t in range(15):
    I = (ts['Frame'] == t) & (ts['FUCCI thresholded'] == 'High')
    mean_vol[t] = ts.loc[I,'Volume'].mean()

plt.plot(mean_vol,'r-')
plt.plot(ts.groupby('Frame').mean()['Volume'],'b-')

plt.legend(['All nuclei population','Nuclei with high FUCCI','All nuclei mean'])

#%%

nuc_vol_normed = np.zeros(len(df))
for i,row in df.iterrows():
   
    nuc_vol_normed[i] = row['Nuclear volume'] / mean_vol[int(row['Frame'])]

df['Nuclear volume normed'] = nuc_vol_normed

#%%

plt.figure(1)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Nuclear volume normed'])

plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume (fL)')

