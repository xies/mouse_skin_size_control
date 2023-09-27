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
# dirnames['Ablation_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'
dirnames['Ablation_R5'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R1'
# dirnames['Ablation_R6'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-31-2023 R26CreER Rb-fl no tam ablation 8hr/F1 Black/R2'
# dirnames['Ablation_R11'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-14-2023 R26CreER Rb-fl no tam ablation 24hr/M5 white/R3'
dirnames['Ablation_R12'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R1/'
# dirnames['Ablation_R13'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/08-23-2023 R26CreER Rb-fl no tam ablation 16h/M5 White DOB 4-25-2023/R2/'

#%%

all_tracks = {}
ts_regions = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['Ablation','Nonablation']:
        
        with open(path.join(dirname,'manual_tracking',f'{name}_{mode}_dense.pkl'),'rb') as file:
            tracks = pkl.load(file)
            
        all_tracks[name+'_'+mode] = tracks
        
        ts_regions[name+'_'+mode] = pd.concat(tracks)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_{mode}_dataframe.csv'),index_col=0)
        # df['Division size'] = df['Birth size'] + df['Total growth']
        # df['S entry size'] = df['Birth size'] + df['G1 growth']
        # df['Log birth size'] = np.log(df['Birth size']) 
        # df['Fold grown'] = df['Division size'] / df['Birth size']
        # df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
ts_all = pd.concat(ts_regions,ignore_index=True)

ablation = ts_all[ts_all['Mode'] == 'Ablation']
nonablation = ts_all[ts_all['Mode'] == 'Nonablation']

#%%

sb.catplot(ts_all,x='Mouse',hue='Mode',y='Specific GR normal',kind='box')
sb.catplot(df_all,x='Mouse',hue='Mode',y='Exponential growth rate',kind='box')
sb.catplot(df_all,x='Mouse',hue='Mode',y='S phase entry size normal',kind='box')

#%%

pairs = zip(range(7),range(1,8))
colors = {'Ablation':'r','Nonablation':'b'}

for first,second in pairs:
    
    for (_,mode),track in ts_all[ts_all['Region'] == 'Ablation_R12'].groupby(['CellID','Mode']):
        plt.subplot(2,4,first+1)
        plt.plot([0,2],[0,2],'k--')
        
        t1 = track[track['Frame'] == first]['Volume normal']
        t2 = track[track['Frame'] == second]['Volume normal']
        if len(t1) == 1 and len(t2) == 1:
            plt.scatter(t1,t2,color=colors[mode],alpha=0.2)
        
    # plt.title(f'{first} v. {second}')
    # plt.xlim([0,2]); plt.ylim([0,2])

#%%

field2plot = 'Volume normal'
YMIN = 0
YMAX = 2
tracks = [v for k,v in ts_all[ts_all['Mode'] == 'Nonablation'].groupby(['Region','CellID'])]

plt.subplot(1,2,1)
for t in tracks:
    plt.plot(t.Age, t[field2plot],'b-',alpha=0.1)
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([YMIN,YMAX])
plt.title('Non neighbors')

tracks = [v for k,v in ts_all[ts_all['Mode'] == 'Ablation'].groupby(['Region','CellID'])]

plt.subplot(1,2,2)
for t in tracks:
    plt.plot(t.Age, t[field2plot],'r-',alpha=0.1)
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
plt.ylim([YMIN,YMAX])
plt.title('Neighbors')

#%%

# ts_all['Specific GR'] = ts_all['Growth rate'] / ts_all['Volume']
# ts_all['Specific GR normal'] = ts_all['Growth rate normal'] / ts_all['Volume normal']
# sb.catplot(ts_all,x='Mode',y='Specific GR normal',kind='box')
sb.catplot(df_all,hue='Mode',y='Exponential growth rate',x='Region',kind='box')

# sb.catplot(ts_all,x='Mode',y='Specific GR normal',kind='box',hue='Region')


plt.figure()
sb.lmplot(ts_all,x='Distance to ablated cell',y='Specific GR normal', scatter_kws={'alpha':.1},hue='Mode')


#%%

from basicUtils import ttest_from_groupby

for mousename,mouse in df_all.groupby('Mouse'):
    
    print(f'--- {mousename} ---')
    # T,P = ttest_from_groupby(mouse,'Mode','S phase entry size')
    # print(f'S phase entry size, P = {P}')
    
    T,P = ttest_from_groupby(mouse,'Mode','S phase entry size normal')
    print(f'S phase entry size normal, P = {P}')
    
    T,P = ttest_from_groupby(mouse,'Mode','Exponential growth rate')
    print(f'Exponential growth rate, P = {P}')


for mousename,mouse in ts_all.groupby('Mouse'):
    print(f'--- {mousename} ---')
    T,P = ttest_from_groupby(mouse,'Mode','Specific GR')
    print(f'Specific GR, P = {P}')
    
    T,P = ttest_from_groupby(mouse,'Mode','Specific GR normal')
    print(f'Specific GR nromal, P = {P}')




