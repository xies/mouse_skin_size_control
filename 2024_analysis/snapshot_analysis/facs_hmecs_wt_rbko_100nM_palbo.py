#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:09:23 2024

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
from mathUtils import cvariation_ci, cvariation_ci_bootstrap

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/07-01-2024 HMECs WT RBKO palbo repeat'

#%%

filelist = glob(path.join(dirname,'*.fcs'))
conditions = ['RBKO palbo','WT palbo','WT','RBKO']
df = []
for i,f in enumerate(filelist):
    _df = FCMeasurement(ID='Test Sample', datafile=filelist[i]).data
    _df['Condition'] = conditions[i]
    df.append(_df)
df = pd.concat(df,ignore_index=True)

#geminin should be on log scale
df['Log-geminin'] = np.log(df['mCherry-A'])

# Gate on FSC-SSC
plt.figure()
pts = plt.scatter(df['FSC-A'],df['SSC-A'],alpha=0.005)
plt.xlabel('FSC-A');plt.ylabel('SSC-A');
selector = SelectFromCollection(plt.gca(), pts)

#%% 

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df['FSC-A'],df['SSC-A'])])

df = df[I]

#%% Gate on singlets based on H/W
plt.figure()
pts = plt.scatter(df['Hoechst-H'],df['SSC-W'],alpha=0.005)
plt.xlabel('Hoechst-height');plt.ylabel('Hoechst-width');
selector = SelectFromCollection(plt.gca(), pts)

#%% 

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

#%% Use High geminin to gate on Gemin+

pts = plt.scatter(diploids['Hoechst-A'],diploids['Log-geminin'],alpha=0.005)
plt.xlabel('Hoechst-area');plt.ylabel('Log-geminin');
selector = SelectFromCollection(plt.gca(), pts)

#%% Geminin-high cells are 'SG2'

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(diploids['Hoechst-A'],diploids['Log-geminin'])])
# I = diploids['Log-geminin'] > th
diploids.loc[:,'Geminin_high'] = False
diploids.loc[I,'Geminin_high'] = True

#%%

Nboot = 1000
CV = pd.DataFrame()

for (condition,phase),_df in diploids.groupby(['Condition','Geminin_high']):
    
    # CV.loc[condition,phase] = stats.variation(_df['Alexa Fluor™ 647-A'])
    CV.loc[condition,phase] = stats.variation(_df['SSC-A'])
    
    # CV.loc[phase,'FSC'] = stats.variation(_df['FSC-A'])
    
    # lb,ub = cvariation_ci(_df['FSC-A'])
    # CV.loc[phase,'FSC lb parametric'],CV.loc[phase,'FSC ub parametric'] = lb,ub
    # CV.loc[phase,'FSC error parametric'] = (ub-lb)/2
    
    # lb,ub = cvariation_ci(_df['SSC-A'])
    # CV.loc[phase,'SSC lb parametric'],CV.loc[phase,'SSC ub parametric'] = lb,ub
    # CV.loc[phase,'SSC error parametric'] = (ub-lb)/2
    
    # lb,ub = cvariation_ci_bootstrap(_df['FSC-A'],Nboot)
    # CV.loc[phase,'FSC lb bootstrap'],CV.loc[phase,'FSC ub bootstrap'] = lb,ub
    # CV.loc[phase,'FSC error bootstrap'] = (ub-lb)/2
    # lb,ub = cvariation_ci_bootstrap(_df['SSC-A'],Nboot)
    # CV.loc[phase,'SSC lb bootstrap'],CV.loc[phase,'SSC ub bootstrap'] = lb,ub
    # CV.loc[phase,'SSC error bootstrap'] = (ub-lb)/2

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
plot_size_CV_subplots(diploids[diploids['Condition'] == 'WT palbo'],'Geminin_high','HMEC WT 100nM Palbo')

plot_size_CV_subplots(diploids[diploids['Condition'] == 'RBKO'],'Geminin_high','HMEC RBKO')
plot_size_CV_subplots(diploids[diploids['Condition'] == 'RBKO palbo'],'Geminin_high','HMEC RBKO Palbo')


