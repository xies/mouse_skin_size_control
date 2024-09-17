#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:26:00 2024

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

from mathUtils import cvariation_ci

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/Yeast/Jordan/230615'

#%%

filelist = glob(path.join(dirname,'*MS358*.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0])

# Gate on singlets
pts = plt.scatter(df['FSC-A'],df['FSC-H'],alpha=0.005)
selector = SelectFromCollection(plt.gca(), pts)

#%%

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['FSC-A'],df['FSC-H'])])

df_ = df[I]

#%% gate on diploids

pts = plt.scatter(df_['YL2-H'],df_['YL2-W'],alpha=.01)
selector = SelectFromCollection(plt.gca(), pts)

#%% Gate diploids

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df_['YL2-H'],df_['YL2-W'])])

diploids = df_[I]

#%% Gate cell cycle based on FUCCI

diploids['Log-PI'] = np.log(diploids['YL2-A'])

# Set Cdt threshold
th = 9.9
plt.hist(diploids['Log-PI'],100);plt.xlabel('Log-PI')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
diploids['High_PI'] = True
diploids.loc[diploids['Log-PI'] < th,'High_PI'] = False

#%%

diploids.loc[~diploids['High_PI'], 'Phase'] = 'G1'
diploids.loc[diploids['High_PI'], 'Phase'] = 'SG2'

(_,g1),(_,sg2) = diploids.groupby('Phase')

#%%

cv_g1,lb_g1,ub_g1 = cvariation_ci(g1['FSC-A'])
cv_sg2,lb_sg2,ub_sg2 = cvariation_ci(sg2['FSC-A'])

print('---- FSC ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {lb_g1:.3}, upper bound: {ub_g1:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {lb_sg2:.3}, upper bound: {ub_sg2:.3}")

cv_g1,lb_g1,ub_g1 = cvariation_ci(g1['SSC-A'])
cv_sg2,lb_sg2,ub_sg2 = cvariation_ci(sg2['SSC-A'])

print('---- SSC ----')
print(f"CV for G1: {cv_g1:.3}, lower bound: {lb_g1:.3}, upper bound: {ub_g1:.3}")
print(f"CV for SG2: {cv_sg2:.3}, lower bound: {lb_sg2:.3}, upper bound: {ub_sg2:.3}")


