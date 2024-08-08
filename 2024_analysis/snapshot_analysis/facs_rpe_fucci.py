#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:55:05 2024

@author: xies
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from FlowCytometryTools import FCMeasurement
from sklearn import mixture
from glob import glob
from os import path

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy import stats
from mathUtils import cvariation_ci, cvariation_bootstrap

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/FlowRepository_FR-FCM-Z3RV_files'

#%%

filelist = glob(path.join(dirname,'RPE-FUCCI cell different serum treatment 1st 12h_10_006.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0]).data

#geminin should be on log scale
df['Log-cdt1'] = np.log(df['Alexa Fluor 488-A'])
df['Log-geminin'] = np.log(df['mCherry-A'])
df['Log-cyan'] = np.log(df['AmCyan-A'])

# Gate on singlets based on H/W
plt.figure()
pts = plt.scatter(df['Log-cdt1'],df['Log-geminin'],alpha=0.005)
selector = SelectFromCollection(plt.gca(), pts)

#%% 

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Log-cdt1'],df['Log-geminin'])])

df.loc[:,'Phase'] = 'NA'
df.loc[I,'Phase'] = 'G1'

# Display gate
plt.figure()
pts = plt.scatter(df['Log-cdt1'],df['Log-geminin'],alpha=0.005)
patch = PathPatch(p_,lw=2,facecolor='r',alpha=0.5)
plt.gca().add_patch(patch)

#%% Use High geminin to gate on Gemin+

pts = plt.scatter(df['Log-cdt1'],df['Log-geminin'],alpha=0.01)
plt.ylim([7,12])
selector = SelectFromCollection(plt.gca(), pts)

#%% Geminin-high cells are 'SG2'

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Log-cdt1'],df['Log-geminin'])])
# I = diploids['Log-geminin'] > th
df.loc[I,'Phase'] = 'G1S'

#%% CVs

(_,g1),(_,g1s),_ = df.groupby('Phase')

cvariation_bootstrap(g1['SSC-A'].values,1000)
cvariation_bootstrap(g1s['SSC-A'].values,1000)

cvariation_bootstrap(g1['FSC-A'].values,1000)
cvariation_bootstrap(g1s['FSC-A'].values,1000)

