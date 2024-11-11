#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:26:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
from skimage import io, measure
from tqdm import tqdm
from scipy.stats import variation
import matplotlib.pyplot as plt
from mathUtils import cvariation_bootstrap, cv_difference_pvalue

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'

#%%

def quantify_volume(f):
    
    im = io.imread(f)
    
    _df = []
    for t in tqdm(range(im.shape[0])):
        this_df = pd.DataFrame(measure.regionprops_table(im[t,...],properties=['label','area']))
        this_df['Frame'] = t
        _df.append(this_df)
    _df = pd.concat(_df)
    _df = _df.rename(columns={'area':'Volume'})
    frames = pd.DataFrame(_df.groupby('label')['Frame'].mean())
    _df = pd.DataFrame(_df.groupby('label')['Volume'].mean())
    _df['Frame'] = frames

    return _df

birth = quantify_volume(path.join(dirname,'Position001_Mastodon/birth/birth_manual.tif'))
birth['Phase'] = 'Birth'
g1s = quantify_volume(path.join(dirname,'Position001_Mastodon/g1s/g1s_manual.tif'))
g1s['Phase'] = 'G1S'
div = quantify_volume(path.join(dirname,'Position001_Mastodon/division/div_manual.tif'))
div['Phase'] = 'Division'
df['Time'] = df['Frame'] / 3
df = pd.concat((birth,g1s,div),ignore_index=True)
df.to_csv(path.join(dirname,'cell_size_by_cellcycle_position.csv'))

#%%

sb.lmplot(df,x='Time',y='Volume',hue='Phase')
plt.xlabel('Time (h)')
plt.ylabel('Nuclear volume at cell cycle phase (h)')

#%% Estimate CV via bootstrap

CV = pd.DataFrame()

for phase,_df in df.groupby('Phase',sort=False):
    cv,lb,ub = cvariation_bootstrap(_df['Volume'].values,Nboot = 10000,subsample=100)
    CV.loc[phase,'CV'] = cv
    CV.loc[phase,'ub'] = ub
    CV.loc[phase,'lb'] = lb
    
plt.errorbar([1,2,3],CV['CV'],yerr = (CV['ub'] - CV['lb'])/2)
plt.ylabel('CV in nuclear size +/- 95% interval via bootstrap')
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])

print(CV)
CV.to_csv(path.join(dirname,'cv_cellcyce_position.csv'))

#%% Calculate P-values via bootstrap

cv_difference_pvalue(df[df['Phase'] == 'Birth']['Volume'],df[df['Phase'] == 'G1S']['Volume'], Nboot=10000)
cv_difference_pvalue(df[df['Phase'] == 'Division']['Volume'],df[df['Phase'] == 'G1S']['Volume'], Nboot=10000)

    

