#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 12:04:02 2021

@author: xies
"""

import numpy as np
from skimage import io, measure
from os import path
import seaborn as sb
from scipy.stats import stats

dirname = '/Users/xies/Box/Mouse/Skin/Two photon/NMS/snapshot_size/segmentation'

dx = 0.2920097

#%%

im_wt = io.imread(path.join(dirname,'wt_day0.tif'))
im_rbko = io.imread(path.join(dirname,'rbko_day0.tif'))

#%%

props = measure.regionprops(im_wt)
wt = pd.DataFrame()
wt['Volume px'] = [ p['Area'] for p in props]
wt['Volume'] = wt['Volume px'] * dx ** 2
wt['Genotype'] = 'WT'

props = measure.regionprops(im_rbko)
rbko = pd.DataFrame()
rbko['Volume px'] = [ p['Area'] for p in props]
rbko['Volume'] = rbko['Volume px'] * dx ** 2
rbko['Genotype'] = 'RBKO'

df = pd.concat((wt,rbko))

#%%

print(df.groupby('Genotype').mean())

sb.catplot(data=df,x='Genotype',y='Volume',kind='boxen')
print(stats.ttest_ind(df.groupby('Genotype')['Volume'].apply(list)['WT'],
                df.groupby('Genotype')['Volume'].apply(list)['RBKO']))

