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
    _ts['Region'] = name
    ts.append(_ts)

df = pd.concat(df,ignore_index=True)
ts = pd.concat(ts,ignore_index=True)

with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c1 = pkl.load(f,encoding='latin-1')
with open('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/2020 CB analysis/exports/collated_manual.pkl','rb') as f:
    c2 = pkl.load(f,encoding='latin-1')
collated = c1+c2
dfc = pd.concat(collated)

#%% Illustrate the problem

plt.figure(1)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Nuclear volume'],'b-',alpha=0.1)
sb.lineplot(data=df,x='Frame',y='Nuclear volume',color='r')
plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume -- cellpose->threshold (fL)')

plt.figure(2)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Nuclear volume raw'],'b-',alpha=0.1)
sb.lineplot(data=df,x='Frame',y='Nuclear volume raw',color='r')
plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume (raw output) (fL)')


# plt.figure(3)
# mesa = dfc.copy()
# mesa['Frame'] = mesa['Frame'] - 1
# for cellID in mesa['CellID'].unique():
#     this_cell = mesa[mesa['CellID'] == cellID]
#     this_cell = this_cell[this_cell['Daughter'] == 'None']
#     plt.plot(this_cell.Frame,this_cell['Nucleus'],'b-',alpha=0.1)
# sb.lineplot(data=mesa[mesa['Daughter'] == 'None'],x='Frame',y='Nucleus',color='r')
# plt.xlabel('Movie frame')
# plt.ylabel('Nuclear volume -- cortical segment -> thresholded (fL)')

plt.figure(4)
for cellID in df['basalID'].unique():
    this_cell = df[df['basalID'] == cellID]
    plt.plot(this_cell.Frame,this_cell['Nuclear volume normalized'],'b-',alpha=0.1)
sb.lineplot(data=df,x='Frame',y='Nuclear volume normalized',color='r')
plt.xlabel('Movie frame')
plt.ylabel('Nuclear volume normalized to per-frame average')


plt.figure(5)
mesa = dfc.copy()
mesa['Frame'] = mesa['Frame'] - 1
for cellID in mesa['CellID'].unique():
    this_cell = mesa[mesa['CellID'] == cellID]
    this_cell = this_cell[this_cell['Daughter'] == 'None']
    plt.plot(this_cell.Frame,this_cell['Volume'],'b-',alpha=0.1)
sb.lineplot(data=mesa[mesa['Daughter'] == 'None'],x='Frame',y='Volume',color='r')
plt.xlabel('Movie frame')
plt.ylabel('Cell volume (fL)')

#%%

import statsmodels.api as sm

df_all = df.rename(columns={'basalID':'CellID'}).merge(mesa,on=['Region','CellID','Frame'])

size_control = []
for cellID in df['basalID'].unique():
    
    this_cell = df_all[df_all['CellID'] == cellID]
    this_cell = this_cell.sort_values('Frame')
    if this_cell.iloc[-1]['Frame'] == 8:
        continue
    
    size_control.append(
        {'Birth nuclear volume raw':this_cell.iloc[0]['Nuclear volume raw']
         ,'Division nuclear volume raw': this_cell.iloc[-1]['Nuclear volume raw']
         ,'Birth nuclear volume':this_cell.iloc[0]['Nuclear volume']
         ,'Division nuclear volume':this_cell.iloc[-1]['Nuclear volume']
         ,'Birth nuclear volume normalized':this_cell.iloc[0]['Nuclear volume normalized']
         ,'Division nuclear volume normalized':this_cell.iloc[-1]['Nuclear volume normalized']
         ,'Birth nuclear volume original':this_cell.iloc[0]['Nucleus']
         ,'Division nuclear volume original':this_cell.iloc[-1]['Nucleus']
         ,'Birth volume':this_cell.iloc[0]['Volume (sm)_y']
         ,'Division volume':this_cell.iloc[-1]['Volume (sm)_y']
            })

size_control = pd.DataFrame(size_control)

size_control['Growth'] = size_control['Division volume'] - size_control['Birth volume']
size_control['Nuclear growth'] = size_control['Division nuclear volume'] - size_control['Birth nuclear volume']
size_control['Nuclear growth original'] = size_control['Division nuclear volume original'] - size_control['Birth nuclear volume original']
size_control['Nuclear growth raw'] = size_control['Division nuclear volume raw'] - size_control['Birth nuclear volume raw']
size_control['Nuclear growth normalized'] = size_control['Division nuclear volume normalized'] - size_control['Birth nuclear volume normalized']

#%%

sb.lmplot(size_control,x='Birth volume',y='Growth')
sb.lmplot(size_control,x='Birth nuclear volume original',y='Nuclear growth original')
sb.lmplot(size_control,x='Birth nuclear volume raw',y='Nuclear growth raw')
sb.lmplot(size_control,x='Birth nuclear volume normalized',y='Nuclear growth normalized')


linreg = sm.OLS(size_control.dropna()['Growth'], sm.add_constant(size_control.dropna()['Birth volume'])).fit()
print(f'Slope from cortical volume = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
linreg = sm.OLS(size_control['Nuclear growth'], sm.add_constant(size_control['Birth nuclear volume'])).fit()
print(f'Slope from nuclear volume = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
linreg = sm.OLS(size_control['Nuclear growth original'], sm.add_constant(size_control['Birth nuclear volume original'])).fit()
print(f'Slope from original = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
linreg = sm.OLS(size_control['Nuclear growth raw'], sm.add_constant(size_control['Birth nuclear volume raw'])).fit()
print(f'Slope from raw = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')
linreg = sm.OLS(size_control['Nuclear growth normalized'], sm.add_constant(size_control['Birth nuclear volume normalized'])).fit()
print(f'Slope from normalized = {linreg.params.values[1]} ({linreg.conf_int().values[1,:]})')



#%%

sb.lineplot(data=df,x='Frame',y='NC ratio')
sb.lineplot(data=df,x='Frame',y='NC ratio raw')
sb.lineplot(data=df,x='Frame',y='NC ratio normalized (sm)')

#%%

sb.lineplot(data=ts,x='Frame',y='Nuclear volume raw')
sb.lineplot(data=ts,x='Frame',y='Nuclear volume')
plt.ylim([50,225])
plt.figure()
sb.lineplot(data=ts[ts['FUCCI thresholded'] == 'High'],x='Frame',y='Nuclear volume')
plt.ylim([50,225])

#%%

df_all = df.rename(columns={'basalID':'CellID'}).merge(mesa,on=['Region','CellID','Frame'])
# df_all = df_all[df_all['Daughter'] == 'None']
sb.lmplot(data = df, x= 'Volume',y='Nuclear volume normalized',hue='Frame',
          facet_kws = {'sharex':True, 'sharey':True},scatter_kws={'alpha':0.1})
# plt.xlim([0,325])
# plt.ylim([0,325])

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

