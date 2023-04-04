#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:06:37 2023

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io,morphology, filters, measure

from os import path
from glob import glob
from tqdm import tqdm

import seaborn as sb

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'

#%%

seg = io.imread(path.join(dirname,'3d_nuc_seg/manual_seg_no_track.tif'))

#%%

_tmp = []
for t in tqdm(range(15)):
    
    this_seg = seg[t,...]
    
    footprint = morphology.cube(5)
    
    this_seg_dilated = morphology.dilation(this_seg,footprint=footprint)
    
    im = io.imread(path.join(dirname,f'im_seq/t{t}.tif'))[...,2]
    
    
    th = filters.threshold_otsu(im)
    im_thresh = im > th
    
    this_seg_dilated[~im_thresh] = 0
    
    
    df = pd.DataFrame(measure.regionprops_table(
        this_seg, properties=['label','area']) )
    df = df.rename(columns={'area':'Volume'})
    
    df_ = pd.DataFrame(measure.regionprops_table(
        this_seg_dilated, properties=['label','area']) )
    
    df_ = df_.rename(columns={'area':'Dilated volume'})
    
    df = df.merge(df_,on='label')
    df['Frame'] = t
    _tmp.append(df)

df = pd.concat(_tmp,ignore_index=True)
df['Volume'] = df['Volume'] * .25**2
df['Dilated volume'] = df['Dilated volume'] * .25**2


#%%

sb.catplot(data=pd.concat(c1,ignore_index=True),x='Frame',y='Nucleus',kind='violin')
sb.catplot(data=df,x='Frame',y='Volume', kind='violin')
sb.catplot(data=df,x='Frame',y='Dilated volume', kind='violin')
sb.lmplot(data=df, x='Volume', y= 'Dilated volume', hue='Frame', scatter_kws={'alpha':0.1})

#%%

mesa = pd.concat(c2,ignore_index=True)
mesa['NC ratio'] = 1/mesa['NC ratio']

sb.catplot(data=mesa, x='Frame', y= 'NC ratio', kind='violin')

