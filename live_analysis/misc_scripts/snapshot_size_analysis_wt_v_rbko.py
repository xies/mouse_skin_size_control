#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:04:02 2021

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from os import path
import seaborn as sb
from scipy.stats import stats

dirname = '/Users/xies/OneDrive - Stanford/Skin/Two photon/NMS/snapshot_size/'

dx = 0.2920097

#%% Load segmentations

seg_wt = io.imread(path.join(dirname,'segmentation','wt_day0.tif'))
seg_rbko = io.imread(path.join(dirname,'segmentation','rbko_day0.tif'))

fucci_wt = io.imread(path.join(dirname,'images','wt_fucci_day0.tif'))
fucci_rbko = io.imread(path.join(dirname,'images','rbko_fucci_day0.tif'))

#%% Threshold FUCCI

wt_th = filters.threshold_niblack(fucci_wt,41)
fucci_wt_th = fucci_wt.copy()
fucci_wt_th[fucci_wt < wt_th] = 0

rbko_th = filters.threshold_niblack(fucci_rbko,41)
fucci_rbko_th = fucci_rbko.copy()
fucci_rbko_th[fucci_rbko < rbko_th] = 0

#%% FUCCI classifier by hand?

wt_g1 = io.imread(path.join(dirname,'segmentation','wt_g1.tif'))
rbko_g1 = io.imread(path.join(dirname,'segmentation','rbko_g1.tif'))

#%%

props = measure.regionprops(wt_g1)
wt = pd.DataFrame()
wt['Volume px'] = [ p['Area'] for p in props]
wt['Volume'] = wt['Volume px'] * dx ** 2
wt['Genotype'] = 'WT'

# props = measure.regionprops(seg_wt,intensity_image = wt_g1)
# wt['FUCCI intensity'] = [ p['mean_intensity'] for p in props ]

props = measure.regionprops(rbko_g1)
rbko = pd.DataFrame()
rbko['Volume px'] = [ p['Area'] for p in props]
rbko['Volume'] = rbko['Volume px'] * dx ** 2
rbko['Genotype'] = 'RBKO'

# props = measure.regionprops(seg_rbko,intensity_image = fucci_rbko_th)
# rbko['FUCCI intensity'] = [ p['mean_intensity'] for p in props ]

df = pd.concat((wt,rbko))


#%%

print(df.groupby('Genotype').mean())

sb.catplot(data=df,x='Genotype',y='Volume',kind='boxen')
print(stats.ttest_ind(df.groupby('Genotype')['Volume'].apply(list)['WT'],
                df.groupby('Genotype')['Volume'].apply(list)['RBKO']))

