#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:38:20 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from skimage import io,measure,util,segmentation
from os import path
from glob import glob
from natsort import natsort
from tqdm import tqdm

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path
from mathUtils import cvariation_ci


dirname = '/Users/xies/Desktop/'

#%%

im = io.imread(path.join(dirname,'fly_gut_snapshot.tif'))

e2f = im[...,0]
cycb = im[...,1]
suh = im[...,2]
h2b = im[...,3]

masks = io.imread(path.join(dirname,'fly_gut_snapshot_h2b_cp_masks.tif'))

#%%

df = pd.DataFrame(measure.regionprops_table(masks,intensity_image=h2b,properties=['label','area','mean_intensity']))
df = df.rename(columns={'mean_intensity':'H2b','area':'Nuclear volume'})

df = df.merge(pd.DataFrame(measure.regionprops_table(masks,intensity_image=e2f,properties=['label','mean_intensity'])))
df = df.rename(columns={'mean_intensity':'E2F'})

df = df.merge(pd.DataFrame(measure.regionprops_table(masks,intensity_image=cycb,properties=['label','mean_intensity'])))
df = df.rename(columns={'mean_intensity':'CycB'})

df = df.merge(pd.DataFrame(measure.regionprops_table(masks,intensity_image=suh,properties=['label','mean_intensity'])))
df = df.rename(columns={'mean_intensity':'SuH'})

df['Log-E2F'] = np.log(df['E2F'])
df['Log-CycB'] = np.log(df['CycB'])
df['Log-SuH'] = np.log(df['SuH'])

df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

#%%

th = -1.3
plt.hist(df['Log-E2F'],30);
plt.vlines(th,ymin=0,ymax=30,color='r')
plt.xlabel('Log-E2F')
df['High_E2F'] = False
df.loc[df['Log-E2F'] > th,'High_E2F'] = True

#%%

th = -1.1
plt.hist(df['Log-CycB'],30);
plt.vlines(th,ymin=0,ymax=30,color='r')
plt.xlabel('Log-CycB')
df['High_CycB'] = False
df.loc[df['Log-CycB'] > th,'High_CycB'] = True

#%%

fucci = df[df['High_E2F'] | df['High_CycB']]




