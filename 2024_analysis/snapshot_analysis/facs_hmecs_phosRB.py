#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:28:19 2024

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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/HMECs/07-30-2024 HMECs WT Hoechst phosRB-488 mCh-Geminin'

#%%

filelist = glob(path.join(dirname,'*.*'))
conditions = ['WT']

df = []
for i,f in enumerate(filelist):
    _df = FCMeasurement(ID='Test Sample', datafile=f).data
    _df['Condition'] = conditions[i]
    df.append(_df)
df = pd.concat(df,ignore_index=True)    

#geminin should be on log scale
df['Log-geminin'] = np.log(df['mCherry-A'])
df['Log-pRB'] = np.log(df['BL1-A'])

# Gate on singlets based on H/W
plt.figure()
pts = plt.scatter(df['Hoechst-H'],df['Hoechst-W'],alpha=0.005)
plt.xlabel('Hoechst-height');plt.ylabel('Hoechst-width');
selector = SelectFromCollection(plt.gca(), pts)

#%%  Gate diploids

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['Hoechst-H'],df['Hoechst-W'])])

diploids = df[I]

# Display gate
plt.figure()
pts = plt.scatter(df['Hoechst-H'],df['Hoechst-W'],alpha=0.005)
plt.xlabel('Hoechst-height');plt.ylabel('Hoechst-width');
patch = PathPatch(p_,lw=2,facecolor='r',alpha=0.5)
plt.gca().add_patch(patch)

#%% Use High geminin to gate on S phase

pts = plt.scatter(diploids['Hoechst-A'],diploids['Log-geminin'],alpha=0.01)
plt.xlabel('Hoechst-A');plt.ylabel('Log-geminin');
selector = SelectFromCollection(plt.gca(), pts)

#%% Geminin-high cells are 'SG2'

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(diploids['Hoechst-A'],diploids['Log-geminin'])])
# I = diploids['Log-geminin'] > th
diploids.loc[:,'Phase'] = 'SG2'
diploids.loc[I,'Phase'] = 'post-G1S'

#%% Use High geminin to gate on S phase

pts = plt.scatter(diploids['Log-pRB'],diploids['Log-geminin'],alpha=0.05)
plt.xlabel('Log-pRB');plt.ylabel('Log-geminin')
selector = SelectFromCollection(plt.gca(), pts)

#%% Geminin-high cells are 'SG2'

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(diploids['Log-pRB'],diploids['Log-geminin'])])
diploids.loc[I,'Phase'] = 'pre-G1'

#%%

Nboot = 1000
CV = pd.DataFrame()

for phase,_df in diploids.groupby('Phase'):
    
    CV.loc[phase,'SSC-A'],_,_ = cvariation_bootstrap(_df['SSC-A'].values,1000)
    CV.loc[phase,'FSC-A'],_,_ = cvariation_bootstrap(_df['FSC-A'].values,1000)
    
print(CV)

#%% Plot the CVs as errorbars

def plot_size_CV_subplots(df,x,title=None):
    
    plt.figure()
    plt.subplot(1,3,1)
    sb.barplot(df,y='FSC-A',x=x
               ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
               ,order=[False,True])
    plt.ylabel('CV of FSC')
    
    plt.subplot(1,3,2)
    sb.barplot(df,y='SSC-A',x=x
               ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
               ,order=[False,True])
    plt.ylabel('CV of SSC')
    plt.title(title)
    
    plt.subplot(1,3,3)
    sb.barplot(df,y='Alexa Fluor™ 647-A',x=x
               ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
               ,order=[False,True])
    plt.ylabel('CV of SE-647')

plot_size_CV_subplots(diploids[diploids['Condition'] == 'WT'],'Geminin_high','HMEC WT')
plot_size_CV_subplots(diploids[diploids['Condition'] == 'WT 50nM Palbo'],'Geminin_high','HMEC WT 50nM Palbo')


