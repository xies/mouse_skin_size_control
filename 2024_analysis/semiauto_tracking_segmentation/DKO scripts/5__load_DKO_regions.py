#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:31:39 2023

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

# dirnames['WT_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
# dirnames['WT_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
# dirnames['WT_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R1'
# dirnames['WT_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M6 WT/R2'

# dirnames['RBKO_R1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
# dirnames['RBKO_R3'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R1'
# dirnames['RBKO_R4'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/03-26-2023 RB-KO pair/M1 RBKO/R2'

# dirnames['RBKOp107het_R2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/05-04-2023 RBKO p107het pair/F8 RBKO p107 het/R2'
dirnames['DKO_R1'] = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Left ear/Post tam/R1/'
dirnames['WT_R1'] = '/Volumes/T7/11-07-2023 DKO/M3 p107homo Rbfl/Right ear/Post Ethanol/R1/'

#%%

from measureSemiauto import recalibrate_pixel_size

all_tracks = {}
all_ts = {}
regions = {}
for name,dirname in dirnames.items():
    for mode in ['curated']:
        
        with open(path.join(dirname,'manual_tracking',f'{name}_complete_cycles_fixed_{mode}.pkl'),'rb') as file:
            tracks = pkl.load(file)
        
        df = pd.read_csv(path.join(dirname,f'manual_tracking/{name}_dataframe_{mode}.csv'),index_col=0)
        
        for t in tracks:
            t['Time to G1/S'] = t['Frame'] - t['S phase entry frame']
            # t['Volume interp'] = smooth_growth_curve
        
        all_tracks[name+'_'+mode] = tracks
        all_ts[name+'_'+mode] = pd.concat(tracks)
        
        df['Division size'] = df['Birth size'] + df['Total growth']
        df['S entry size'] = df['Birth size'] + df['G1 growth']
        df['Log birth size'] = np.log(df['Birth size']) 
        df['Fold grown'] = df['Division size'] / df['Birth size']
        df['SG2 growth'] = df['Total growth'] - df['G1 growth']
        regions[name+'_'+mode] = df

df_all = pd.concat(regions,ignore_index=True)
all_ts = pd.concat(all_ts,ignore_index=True)

wt = df_all[df_all['Genotype'] == 'WT']
dko = df_all[df_all['Genotype'] == 'DKO']

#%%

for t in all_tracks['WT_R1_curated']:
    plt.plot(t.Frame,t.Volume)

#%%

sb.lmplot(df_all,x='Birth size',y='G1 growth',col='Mode',row='Pair',hue='Genotype')

#%%

sb.lmplot(df_all,x='Birth size',y='G1 length',row='Pair',
          robust=False,hue='Genotype')
plt.xlim([160,550])
plt.ylim([0,100])

#%%

g = sb.FacetGrid(df_all, hue="Genotype", col ='Pair')
g.map(sb.histplot,'G1 length')
plt.legend()

#%%

sb.catplot(df_all,x='Pair',y='S phase entry size normal',hue='Genotype',kind='violin')

sb.catplot(df_all,x='Pair',y='Birth size normal',hue='Genotype',kind='violin')

#%%

sb.relplot(all_ts,x='Age',y='Volume',col='Pair',row='Mode',hue='Genotype',kind='line')

#%%

dko_first_gen = dko[dko['Generation'] == 1]
dko_second_gen = dko[dko['Generation'] == 2]

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
        print(x)
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    p = np.polyfit(X,Y,1)
    return p

def plot_reg_with_bin(x,y,data=None,alpha=0.5,bin_numbers=8):
    if not data is None:
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    sb.regplot(x=x,y=y,scatter_kws={'alpha':alpha},robust=True)
    plot_bin_means(x,y,minimum_n=4,bin_edges=bin_numbers,bin_style='equal')



#%%

for mouse in ['WT_M1','WT_M3','RBKO_M2','RBKO_M4']:

    df_ = df_curated[df_curated['Mouse'] == mouse]
    p = nonan_lin_reg(df_['Birth size normal'],df_['G1 growth normal'])
    print(f'{mouse} = {p}')




