#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:27:58 2024

@author: xies
"""

from skimage import io, measure
import pandas as pd
from os import path
from glob import glob

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/Confocal/2019/03-05-2019 Lgr5GFP FUCCI G1 500nM Palbo/DMSO 24hr/DAPI Lgr5GFP FUCCI-G1-mKO betaCat647/'

filelist = glob(path.join(dirname,'*_segmentations.tif'))

df = []
for f in filelist[:2]:
    
    labels = io.imread(f)
    
    _df = pd.DataFrame(measure.regionprops_table(labels[:,1,...],properties=['label','area']))
    _df = _df.rename(columns={'area':'Cell volume'})
    _df = _df.merge(pd.DataFrame(measure.regionprops_table(labels[:,2,...],properties=['area','label']))
                   ,on='label')
    _df = _df.rename(columns={'area':'Nuclear volume'})
    _df['Region'] = f
    df.append(_df)
    
df = pd.concat(df,ignore_index=True)

df['Cell volume'] = df['Cell volume'] * 0.1317882 **2
df['Nuclear volume'] = df['Nuclear volume'] * 0.1317882 **2

#%%

sb.lmplot(df,x='Cell volume',y='Nuclear volume')

plt.scatter([0],[0])

