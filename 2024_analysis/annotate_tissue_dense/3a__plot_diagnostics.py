#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:25:42 2024

@author: xies
"""

import numpy as np
# from skimage import io, measure, draw, util, morphology
import pandas as pd

from basicUtils import nonan_pairs

import matplotlib.pylab as plt
import seaborn as sb

from tqdm import tqdm
from os import path

dx = 0.25
XX = 460
Z_SHIFT = 10

# Differentiating thresholds
centroid_height_cutoff = 3.5 #microns above BM

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2'

df = pd.read_csv(path.join(dirname,'tissue_dataframe.csv'),index_col=0)

#%% Centroids should be VERY close b/w nuc + cyto --> R >> 0.9

for t,df_t in df.groupby('Frame'):
    
    X,Y = nonan_pairs(df_t['X-cell'],df_t['X'])
    R,P = np.corrcoef(X,Y)
    print(f'Frame {t}: X coord -- R2 = {R**2}')
    
    X,Y = nonan_pairs(df_t['Z-cell'],df_t['Z'])
    R,P = np.corrcoef(X,Y)
    print(f'Frame {t}: Z coord -- R2 = {R**2}')
    
#%% Plot the different cortical segmentation features and how they relate to nuclear features

sb.catplot(df,x='Frame',y='NC ratio',kind='violin')
# plt.figure()
sb.lmplot(df,x ='Nuclear volume',y='Cell volume',hue='Frame', palette='Set2')
sb.lmplot(df,x='Apical area',y='Basal area')
sb.lmplot(df,x='Cell volume',y='Apical area')
sb.lmplot(df,x='Cell volume',y='Middle area')
sb.lmplot(df,x='Cell volume',y='Basal area')

plt.subplot(2,1,1)
sb.histplot(df,x='Nuclear volume',y='Frame',bins=[15,15])
plt.subplot(2,1,2)
sb.histplot(df,x='Cell volume',y='Frame',bins=[15,15])

#%%


