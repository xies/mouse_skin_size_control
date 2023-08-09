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

# dirnames['Ablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Ablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
dirnames['Ablation_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'
# dirnames['Nonablation_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-23-2023 R26CreER Rb-fl no tam ablation/R1/'
dirnames['Nonablation_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R1'
dirnames['Nonablation_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/07-26-2023 R25CreER Rb-fl no tam ablation 12h/Black female/R2'

#%%

all_tracks = {}
ts_all = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['curated']:
        
        with open(path.join(dirname,'manual_tracking',f'{name}_dense_{mode}.pkl'),'rb') as file:
            tracks = pkl.load(file)
        
        for t in tracks:
            t['Time to G1/S'] = t['Frame'] - t['S phase entry frame']
            # t['Volume interp'] = smooth_growth_curve
            
        all_tracks[name+'_'+mode] = tracks
        
        ts_all[name+'_'+mode] = pd.concat(tracks)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'),index_col=0)
        df['Division size'] = df['Birth size'] + df['Total growth']
        df['S entry size'] = df['Birth size'] + df['G1 growth']
        df['Log birth size'] = np.log(df['Birth size']) 
        df['Fold grown'] = df['Division size'] / df['Birth size']
        df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
ts_all = pd.concat(ts_all,ignore_index=True)

#%%

plt.subplot(1,2,1)
for t in all_tracks['Nonablation_R4_curated']:
    plt.plot(t.Age, t['Volume normal interp'],'b-')
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
# plt.ylim([0.5,2.5])
plt.title('Non neighbors')
    
plt.subplot(1,2,2)
for t in all_tracks['Ablation_R4_curated']:
    plt.plot(t.Age, t['Volume normal interp'],'r-')
plt.xlabel('Time since ablation (h)')
plt.ylabel('Volume (px)')
# plt.ylim([0.5,2.5])
plt.title('Neighbors')
    
# sb.relplot(ts_all,x='Age',y='Volume normal',hue='Region', kind='line')

#%%

ts_all['Specific GR'] = ts_all['Growth rate'] / ts_all['Volume']
ts_all['Specific GR normal'] = ts_all['Growth rate normal'] / ts_all['Volume normal']

sb.catplot(ts_all,x='Region',y='Specific GR',kind='violin')

ablation_coords = pd.read_csv(path.join(dirnames['Ablation_R3'],'manual_tracking/ablation_xyz.csv')
                              ,index_col=0,names=['T','Z','Y','X'],header=0)

def find_closest_ablation(df,ablations):
    Ncells = len(df)
    Nablations = len(ablations)
    D = np.zeros((Ncells,Nablations))
    for i in range(Nablations):
        abl = ablations.iloc[i]
        dx = df['X'] - abl['X']
        dy = df['Y'] - abl['Y']
        
        D[:,i] = dx**2 + dy**2
    return D.min(axis=1)

plt.figure()
ts_all['Distance to ablation'] = find_closest_ablation(ts_all,ablation_coords)
sb.regplot(ts_all,x='Distance to ablation',y='Specific GR')


