#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 17:05:27 2022

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
dirnames['WT_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
dirnames['WT_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
dirnames['WT_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R1'
# dirnames['WT_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'

dirnames['RBKO_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
dirnames['RBKO_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R1'
dirnames['RBKO_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R2'

#%%

all_tracks = {}
all_ts = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['curated','manual']:
        
        with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'rb') as file:
            tracks = pkl.load(file)
        
        for t in tracks:
            t['Time to G1/S'] = t['Frame'] - t['S phase entry frame']
            # t['Volume interp'] = smooth_growth_curve
            t['Genotype'] = name.split('_')[0]
            t['Region'] = name
            t['Mouse'] = mouse[name]
            t['Pair'] = pairs[mouse[name]]
            t['Volume'] = t['Volume'] / dx[name]**2
            
        all_tracks[name+'_'+mode] = tracks
        
        all_ts[name+'_'+mode] = pd.concat(tracks)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'),index_col=0)
        df['Division size'] = df['Birth size'] + df['Total growth']
        df['S entry size'] = df['Birth size'] + df['G1 growth']
        df['Log birth size'] = np.log(df['Birth size']) 
        df['Fold grown'] = df['Division size'] / df['Birth size']
        df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        df['Mouse'] = mouse[name]
        df['Pair'] = pairs[mouse[name]]
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
all_ts = pd.concat(all_ts,ignore_index=True)

wt = df_all[df_all['Genotype'] == 'WT']
wt_curated = wt[wt['Mode'] == 'curated']
rbko = df_all[df_all['Genotype'] == 'RBKO']
rbko_curated = rbko[rbko['Mode'] == 'curated']

sb.lmplot(df_all,x='Birth size normal',y='G1 growth normal',col='Mode',row='Pair',hue='Genotype')

#%%

color = {'WT':'r','RBKO':'b'}
# for tracks in all_tracks[1]:
tracks = all_tracks['WT_R2_curated']
for t in tracks:
    plt.plot(t.Age,t['Volume normal'],color=color[t.iloc[0]['Genotype']],alpha=0.1)

#%%

sb.relplot(all_ts,x='Age',y='Volume normal',col='Pair',row='Mode',hue='Genotype',kind='line')


#%%

def nonan_pearson_R(x,y,data=None):
    from basicUtils import nonan_pairs
    if not data is None:
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    R = np.corrcoef(X,Y)
    return R[0,1]


def nonan_lin_reg(x,y,data=None):
    from basicUtils import nonan_pairs
    if not data is None:
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    p = np.polyfit(X,Y,1)
    return p

def plot_reg_with_bin(x,y,data=None,alpha=0.5):
    if not data is None:
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    sb.regplot(x=x,y=y,scatter_kws={'alpha':alpha},robust=True)
    plot_bin_means(x,y,minimum_n=4,bin_edges=5,bin_style='equal')


#%%

plt.subplot(2,2,1)
X,Y = nonan_pairs(wt_curated['Birth size normal'].astype(float),wt_curated['G1 length'].astype(float))
Y = jitter(Y,.2)
plt.xlim([0,2]); plt.title('Wild type')
plot_reg_with_bin(X,Y); plt.ylabel('G1 length (h)')


plt.subplot(2,2,2)
X,Y = nonan_pairs(rbko_curated['Birth size normal'].astype(float),rbko_curated['G1 length'].astype(float))
Y = jitter(Y,.2)
plt.xlim([0,2]); plt.title('RB1 -/-')
plot_reg_with_bin(X,Y); plt.ylabel('G1 length (h)')


plt.subplot(2,2,3)
X,Y = nonan_pairs(wt_curated['Birth size normal'].astype(float),wt_curated['SG2 length'].astype(float))
Y = jitter(Y,.2)
plt.xlim([0,2]); plt.title('Wild type')
plot_reg_with_bin(X,Y); plt.ylabel('SG2 length (h)')


plt.subplot(2,2,4)
X,Y = nonan_pairs(rbko_curated['Birth size normal'].astype(float),rbko_curated['SG2 length'].astype(float))
Y = jitter(Y,.2)
plt.xlim([0,2]); plt.title('RB1 -/-')
plot_reg_with_bin(X,Y); plt.ylabel('SG2 length (h)')

# X,Y = nonan_pairs(regions[2]['Log birth size'].astype(float),regions[2]['G1 length'].astype(float))
# plot_reg_with_bin(X,Y)

#%%

plt.figure()
plt.subplot(2,1,1)
X,Y = nonan_pairs(wt_curated['Birth size normal'].astype(float),wt_curated['G1 growth normal'].astype(float))
plot_reg_with_bin(X,Y)
plt.title('WT')

plt.subplot(2,1,2)
X,Y = nonan_pairs(rbko_curated['Birth size normal'].astype(float),rbko_curated['G1 growth normal'].astype(float))
plot_reg_with_bin(X,Y)
plt.title('RBKO')

# plt.legend(list(dirnames.keys()))





