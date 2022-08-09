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
from ifUtils import *

# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-14-2022 beamexpander test/manual'
# dirname = '/Users/xies/OneDrive - Stanford/Skin/06-25-2022/manual_seg'
dirnames = []
dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Confocal/07-19-2022 Skin/DS 06-25-22 H2B-Cerulean FUCCI2 Phall647')
dirnames.append('/Users/xies/OneDrive - Stanford/Skin/Confocal/07-19-2022 Skin/DS 06-25-22 H2BCerulean FUCCI2 Phall647 second')

#%%

h2b = []
fucci = []
labels = []


# filelist = glob(path.join(dirname,'h2b/*.tif'))
# h2b = list(map(io.imread,filelist))

# filelist = glob(path.join(dirname,'fucci/*.tif'))
# fucci = list(map(io.imread,filelist))

# filelist = glob(path.join(dirname,'labels/*.tif'))
# labels = list(map(io.imread,filelist))

h2b = [io.imread(path.join(dirnames[1],'WT1/WT1.tif')),
        io.imread(path.join(dirnames[0],'RBKO5/RBKO5.tif'))]

labels = [io.imread(path.join(dirnames[1],'WT1/WT1_labels.tif')),
        io.imread(path.join(dirnames[0],'RBKO5/RBKO5_labels.tif'))]

fucci = [io.imread(path.join(dirnames[1],'WT1/WT1_fucci.tif')),
        io.imread(path.join(dirnames[0],'RBKO5/RBKO5_fucci.tif'))]

dataset = list(zip(h2b,fucci,labels))
# dataset = list(zip(h2b,labels))

names = ['WT','RBKO']
dx = [1, 1]

labels = [delete_border_objects(l) for l in labels]

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
    df_['FUCCI norm'] = standardize(df_['FUCCI'])
    # Normalize mean intensities
    df_['H2b norm'] = standardize(df_['H2b'])
    df_['Volume'] = df_['Volume'] * dx[i]
    
    regions.append(df_)

df = pd.concat(regions,ignore_index=True)
df = df[df['Volume'] > 100]

#%% plotting

# Size v. cell cycle
# sb.lmplot(data=df,hue='Genotype',x='FUCCI norm',y='Volume',fit_reg=False)
sb.lmplot(data=df,hue='Genotype',x='FUCCI norm',y='Volume', fit_reg=False)
sb.catplot(data=df,x='Genotype',y='Volume')

print(df.groupby('Genotype').mean()['Volume'])

#%% 

df.to_csv(path.join(dirname,'cell_size.csv'))


