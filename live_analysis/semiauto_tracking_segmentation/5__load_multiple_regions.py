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
# dirnames['WT1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
# dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R2'
# dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M1 WT/R1'


dirnames['RBKO1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
# dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R2'
# dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M6 RBKO/R1'

genotypes = {'WT1':'WT','WT2':'WT','RBKO1':'RBKO','RBKO2':'RBKO'}


#%%

all_tracks = []
all_ts = []
regions = []
for name,dirname in dirnames.items():
    
    with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
        tracks = pkl.load(file)
        
    all_tracks.append(tracks)
    
    tracks = pd.concat(tracks)
    tracks['Genotype'] = genotypes[name]
    tracks['Region'] = name
        
    all_ts.append(tracks)
    
    df = pd.read_csv(path.join(dirname,'dataframe.csv'),index_col=0)
    df['Division size'] = df['Birth size'] + df['Total growth']
    df['S entry size'] = df['Birth size'] + df['G1 growth']
    df['Log birth size'] = np.log(df['Birth size'])
    regions.append(df)

df_all = pd.concat(regions)
all_ts = pd.concat(all_ts)

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

def plot_reg_with_bin(x,y,data=None):
    if not data is None:
        x = data[x]
        y = data[y]
    X,Y = nonan_pairs(x,y)
    sb.regplot(x=x,y=y)
    plot_bin_means(x,y,minimum_n=4,bin_edges=8,bin_style='equal')


#%%

plt.subplot(2,1,1)
X,Y = nonan_pairs(regions[0]['Log birth size'].astype(float),regions[0]['G1 length'].astype(float))
plt.xlim([4,5.6]); plt.title('Wild type')
plot_reg_with_bin(X,Y); plt.ylabel('G1 length (h)')


plt.subplot(2,1,2)
regions[1] = regions[1][regions[1]['Birth frame'] != 2]
X,Y = nonan_pairs(regions[1]['Log birth size'].astype(float),regions[1]['G1 length'].astype(float))
plt.xlim([4,5.6]); plt.title('RB1 -/-')
plot_reg_with_bin(X,Y); plt.ylabel('G1 length (h)')

# X,Y = nonan_pairs(regions[2]['Log birth size'].astype(float),regions[2]['G1 length'].astype(float))
# plot_reg_with_bin(X,Y)

#%%
plt.figure()
X,Y = nonan_pairs(regions[0]['Birth size'].astype(float),regions[0]['G1 growth'].astype(float))
plot_reg_with_bin(X,Y)

X,Y = nonan_pairs(regions[1]['Birth size'].astype(float),regions[1]['G1 growth'].astype(float))
plot_reg_with_bin(X,Y)



plt.legend(list(dirnames.keys()))




