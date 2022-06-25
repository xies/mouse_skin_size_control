#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 18:26:39 2022

@author: xies
"""

import numpy as np
import pandas as pd
from skimage import io, measure
from glob import glob
from os import path
import seaborn as sb
import matplotlib.pyplot as plt

dirname = '/Users/xies/OneDrive - Stanford/Skin/06-14-2022 beamexpander test/manual'

#%%

filelist = sorted(glob(path.join(dirname,'h2b/*.tif')))
h2bs = list(map(io.imread,filelist))
filelist = sorted(glob(path.join(dirname,'fucci/*.tif')))
fuccis = list(map(io.imread,filelist))
filelist = sorted(glob(path.join(dirname,'labels/*.tif')))
labels = list(map(io.imread,filelist))

dataset = list(zip(h2bs,fuccis,labels))

names = ['RBKO','WT']
dx = [0.19, 0.387]

#%% Measure and collate

def standardize(X):
    return (X - X.mean()) / X.std()

regions = []

for i,(h2b,fucci,label) in enumerate(dataset):
    
    h2b_table = measure.regionprops_table(label,h2b,
                                            properties = ['label','centroid','area','mean_intensity'])
    fucci_table = measure.regionprops_table(label,fucci,
                                            properties = ['mean_intensity'])
    
    df_ = pd.DataFrame(h2b_table)
    df_['Genotype'] = names[i]
    df_ = df_.rename(columns={'mean_intensity':'H2b'
                            ,'centroid-0':'Z'
                            ,'centroid-1':'X'
                            ,'centroid-2':'Y'
                            ,'area':'Volume'
                            })
    df_['FUCCI'] = fucci_table['mean_intensity']
    # Normalize mean intensities
    df_['H2b norm'] = standardize(df_['H2b'])
    df_['FUCCI norm'] = standardize(df_['FUCCI'])
    
    df_['Volume'] = df_['Volume'] * dx[i]
    
    regions.append(df_)

df = pd.concat(regions,ignore_index=True)

    
#%% plotting

# Size v. cell cycle
sb.lmplot(data=df,hue='Genotype',x='Volume',y='FUCCI norm', fit_reg=False)

#%% 



