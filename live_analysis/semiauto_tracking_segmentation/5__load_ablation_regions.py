#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 20:06:53 2023

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from skimage import io, measure
import seaborn as sb
from os import path
from glob import glob
from tqdm import tqdm

import pickle as pkl

from basicUtils import *

dirnames = {}

dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Ablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
dirnames['Ablation_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'
dirnames['Ablation_R5'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1'
# dirnames['Ablation_R6'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'

#%%

all_tracks = {}
ts_all = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['Ablation','Nonablation']:
        
        with open(path.join(dirname,'manual_tracking',f'{name}_{mode}_dense.pkl'),'rb') as file:
            tracks = pkl.load(file)
            
        all_tracks[name+'_'+mode] = tracks
        
        ts_all[name+'_'+mode] = pd.concat(tracks)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_{mode}_dataframe.csv'),index_col=0)
        # df['Division size'] = df['Birth size'] + df['Total growth']
        # df['S entry size'] = df['Birth size'] + df['G1 growth']
        # df['Log birth size'] = np.log(df['Birth size']) 
        # df['Fold grown'] = df['Division size'] / df['Birth size']
        # df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
ts_all = pd.concat(ts_all,ignore_index=True)

ablation = ts_all[ts_all['Mode'] == 'Ablation']
nonablation = ts_all[ts_all['Mode'] == 'Nonablation']

#%%

plt.subplot(1,2,1)
for t in all_tracks['Ablation_R5_Nonablation']:
    plt.plot(t.Age, t['Volume normal interp'],'b-')
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([0.5,2.5])
# plt.ylim([0,200])
plt.title('Non neighbors')

plt.subplot(1,2,2)
for t in all_tracks['Ablation_R5_Ablation']:
    plt.plot(t.Age, t['Volume normal interp'],'r-')
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([0.5,2.5])
# plt.ylim([0,200])
plt.title('Neighbors')

#%%

# ts_all['Specific GR'] = ts_all['Growth rate'] / ts_all['Volume']
# ts_all['Specific GR normal'] = ts_all['Growth rate normal'] / ts_all['Volume normal']

sb.catplot(ts_all,x='Mode',y='Specific GR normal',kind='box',hue='Region')


plt.figure()
sb.lmplot(ts_all,x='Distance to ablated cell',y='Specific GR normal', scatter_kws={'alpha':.1},hue='Mode')


#%%

from basicUtils import ttest_from_groupby

T,P = ttest_from_groupby(df_all,'Mode','S phase entry size')
print(f'S phase entry size, P = {P}')

T,P = ttest_from_groupby(df_all,'Mode','S phase entry size normal')
print(f'S phase entry size normal, P = {P}')


T,P = ttest_from_groupby(ts_all,'Mode','Specific GR')
print(f'Specific GR, P = {P}')

T,P = ttest_from_groupby(ts_all,'Mode','Specific GR normal')
print(f'Specific GR nromal, P = {P}')




