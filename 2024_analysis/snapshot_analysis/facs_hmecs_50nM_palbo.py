#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:29:40 2024

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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/06-21-2024 HMECs RB WT 50nM Palbociclib mCh-Geminin'

#%%

filelist = glob(path.join(dirname,'HMECs WT mCh-Geminin Fixed Hoechst SE647.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0]).data

#geminin should be on log scale
df['Log-geminin'] = np.log(df['mCherry-A'])

# Gate on singlets based on H/W
plt.figure()
pts = plt.scatter(df['Hoechst-H'],df['Hoechst-W'],alpha=0.005)
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
CV = pd.DataFrame(index=[True,False])

for phase,_df in diploids.groupby('Geminin_high'):
    
    CV.loc[phase,'SSC'] = stats.variation(_df['SSC-A'])
    CV.loc[phase,'FSC'] = stats.variation(_df['FSC-A'])
    
    lb,ub = cvariation_ci(_df['FSC-A'])
    CV.loc[phase,'FSC lb parametric'],CV.loc[phase,'FSC ub parametric'] = lb,ub
    CV.loc[phase,'FSC error parametric'] = (ub-lb)/2
    
    lb,ub = cvariation_ci(_df['SSC-A'])
    CV.loc[phase,'SSC lb parametric'],CV.loc[phase,'SSC ub parametric'] = lb,ub
    CV.loc[phase,'SSC error parametric'] = (ub-lb)/2
    
    lb,ub = cvariation_ci_bootstrap(_df['FSC-A'],Nboot)
    CV.loc[phase,'FSC lb bootstrap'],CV.loc[phase,'FSC ub bootstrap'] = lb,ub
    CV.loc[phase,'FSC error bootstrap'] = (ub-lb)/2
    lb,ub = cvariation_ci_bootstrap(_df['SSC-A'],Nboot)
    CV.loc[phase,'SSC lb bootstrap'],CV.loc[phase,'SSC ub bootstrap'] = lb,ub
    CV.loc[phase,'SSC error bootstrap'] = (ub-lb)/2


#%% Plot the CVs as errorbars

plt.figure()
plt.subplot(1,3,1)
sb.barplot(diploids,y='FSC-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of FSC')
plt.ylim([0,.25])
plt.title('HMECs WT cell cycle determined by High/Low Geminin')

plt.subplot(1,3,2)
sb.barplot(diploids,y='SSC-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of SSC')
plt.ylim([0,.7])

plt.subplot(1,3,3)
sb.barplot(diploids,y='Alexa Fluor™ 647-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of SE-647')
plt.ylim([0,.7])

# #%% Gate only using DAPI

# th = 0.44e6
# # model = mixture.GaussianMixture(n_components=2).fit(diploids['Hoechst-A'].values.reshape(-1,1))

# plt.hist(diploids['Hoechst-A'],100);plt.xlabel('Hoechst-A')
# plt.vlines(x=th,ymin=0,ymax=1000,color='r')
# diploids['High_DAPI'] = True
# diploids.loc[diploids['Hoechst-A'] < th,'High_DAPI'] = False

# #%%

# CV_dapi = pd.DataFrame()
# CV_dapi['FSC'] = diploids.groupby('High_DAPI')['FSC-A'].apply(stats.variation)
# CV_dapi['SSC'] = diploids.groupby('High_DAPI')['SSC-A'].apply(stats.variation)
# CV_dapi['SE647'] = diploids.groupby('High_DAPI')['Alexa Fluor™ 647-A'].apply(stats.variation)

#%%

filelist = glob(path.join(dirname,'HMECs WT mCh-Geminin 50nM Palbociclib Fixed Hoechst SE647.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0]).data

#geminin should be on log scale
df['Log-geminin'] = np.log(df['mCherry-A'])

# Gate on singlets based on H/W
plt.figure()
pts = plt.scatter(df['Hoechst-H'],df['Hoechst-W'],alpha=0.005)
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
CV = pd.DataFrame(index=[True,False])

for phase,_df in diploids.groupby('Geminin_high'):
    
    CV.loc[phase,'SSC'] = stats.variation(_df['SSC-A'])
    CV.loc[phase,'FSC'] = stats.variation(_df['FSC-A'])
    
    lb,ub = cvariation_ci(_df['FSC-A'])
    CV.loc[phase,'FSC lb parametric'],CV.loc[phase,'FSC ub parametric'] = lb,ub
    CV.loc[phase,'FSC error parametric'] = (ub-lb)/2
    
    lb,ub = cvariation_ci(_df['SSC-A'])
    CV.loc[phase,'SSC lb parametric'],CV.loc[phase,'SSC ub parametric'] = lb,ub
    CV.loc[phase,'SSC error parametric'] = (ub-lb)/2
    
    lb,ub = cvariation_ci_bootstrap(_df['FSC-A'],Nboot)
    CV.loc[phase,'FSC lb bootstrap'],CV.loc[phase,'FSC ub bootstrap'] = lb,ub
    CV.loc[phase,'FSC error bootstrap'] = (ub-lb)/2
    lb,ub = cvariation_ci_bootstrap(_df['SSC-A'],Nboot)
    CV.loc[phase,'SSC lb bootstrap'],CV.loc[phase,'SSC ub bootstrap'] = lb,ub
    CV.loc[phase,'SSC error bootstrap'] = (ub-lb)/2


#%% Plot the CVs as errorbars

plt.figure()
plt.subplot(1,3,1)
sb.barplot(diploids,y='FSC-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of FSC')
plt.ylim([0,.25])
plt.title('HMECs 50nM Palbo, cell cycle determined by High/Low Geminin')

plt.subplot(1,3,2)
sb.barplot(diploids,y='SSC-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of FSC')
plt.ylim([0,.7])

plt.subplot(1,3,3)
sb.barplot(diploids,y='Alexa Fluor™ 647-A',x='Geminin_high'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=[False,True])
plt.ylabel('CV of SE-647')
plt.ylim([0,.7])

#%% Gate only using DAPI

# th = 0.44e6
# # model = mixture.GaussianMixture(n_components=2).fit(diploids['Hoechst-A'].values.reshape(-1,1))

# plt.hist(diploids['Hoechst-A'],100);plt.xlabel('Hoechst-A')
# plt.vlines(x=th,ymin=0,ymax=1000,color='r')
# diploids['High_DAPI'] = True
# diploids.loc[diploids['Hoechst-A'] < th,'High_DAPI'] = False

# #%%

# CV_dapi = pd.DataFrame()
# CV_dapi['FSC'] = diploids.groupby('High_DAPI')['FSC-A'].apply(stats.variation)
# CV_dapi['SSC'] = diploids.groupby('High_DAPI')['SSC-A'].apply(stats.variation)
# CV_dapi['SE647'] = diploids.groupby('High_DAPI')['Alexa Fluor™ 647-A'].apply(stats.variation)



