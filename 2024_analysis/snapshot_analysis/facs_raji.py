#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:29:07 2024

@author: xies
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from FlowCytometryTools import FCMeasurement
from glob import glob
from os import path

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path

from scipy import stats
from mathUtils import cvariation_ci, cvariation_ci_bootstrap

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow FUCCI/06-04-2024 Raji Hoechst CV'

#%%

filelist = glob(path.join(dirname,'*Raji*.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['VL1-H'],df['VL1-W'],alpha=0.005)
plt.xlabel('DAPI-height');plt.ylabel('DAPI-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['VL1-H'],df['VL1-W'])])

df_ = df[I]

#%% gate on diploids

pts = plt.scatter(df_['FSC-A'],df_['FSC-H'],alpha=.01)
plt.xlabel('FSC-A');plt.ylabel('FSC-h');
selector = SelectFromCollection(plt.gca(), pts)

#%% Gate against Cdt1+ 4n cells

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df_['FSC-A'],df_['FSC-H'])])

diploids = df_[I]

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.44e6
plt.hist(diploids['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
diploids['High_DAPI'] = True
diploids.loc[diploids['VL1-A'] < th,'High_DAPI'] = False

(_,twoN),(_,fourN) = diploids.groupby('High_DAPI')

#%% Plot the CVs as errorbars
Nboot = 100

sb.barplot(diploids,y='FSC-A',x='High_DAPI'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           )
plt.ylabel('CV of FSC')
plt.ylim([0,.25])
plt.title('Raji, cell cycle determined by DAPI')



