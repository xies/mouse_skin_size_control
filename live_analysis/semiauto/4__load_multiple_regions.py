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


dirnames = {}
dirnames['WT1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/WT/R1'
dirnames['WT2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M1 WT/R1'

dirnames['RBKO1'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/09-29-2022 RB-KO pair/RBKO/R1'
dirnames['RBKO2'] = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/06-25-2022/M6 RBKO/R1'

#%%

# all_tracks = []

regions = []
for name,dirname in dirnames.items():
    
    # with open(path.join(dirname,'manual_tracking','complete_cycles_fixed.pkl'),'rb') as file:
    #     tracks = pkl.load(file)
    regions.append(pd.read_csv(path.join(dirname,'dataframe.csv'),index_col=0))

df_all = pd.concat(regions)

#%%

X,Y = nonan_pairs(regions[0]['Birth size'].astype(float),regions[0]['G1 growth'].astype(float))
plot_bin_means(X,Y,minimum_n=3,bin_edges=6,bin_style='percentile')
X,Y = nonan_pairs(regions[1]['Birth size'].astype(float),regions[1]['G1 growth'].astype(float))
plot_bin_means(X,Y,minimum_n=3,bin_edges=6,bin_style='percentile')

X,Y = nonan_pairs(regions[2]['Birth size'].astype(float),regions[2]['G1 growth'].astype(float))
plot_bin_means(X,Y,minimum_n=3,bin_edges=6,bin_style='percentile')
X,Y = nonan_pairs(regions[3]['Birth size'].astype(float),regions[3]['G1 growth'].astype(float))
plot_bin_means(X,Y,minimum_n=3,bin_edges=6,bin_style='percentile')

plt.legend(list(dirnames.keys()))