#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:10:15 2024

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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/SZ-062821 Fucci2 P20 F medium test'

#%%

filelist = glob(path.join(dirname,'*Complete*.fcs'))
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

plt.figure()
pts = plt.scatter(df_['VL1-A'],df_['YL2-A'],alpha=.01)
plt.xlabel('DAPI-area');plt.ylabel('Cdt1-area');
selector = SelectFromCollection(plt.gca(), pts)

#%% Gate against Cdt1+ 4n cells

verts = np.array(selector.poly.verts)
x = verts[:,0];y = verts[:,1]
p_ = Path(np.array([x,y]).T)
I = np.array([p_.contains_point([x,y]) for x,y in zip(df_['VL1-H'],df_['VL1-W'])])

diploids = df_[I]

#%% Gate cell cycle based on FUCCI

diploids['Log-Cdt'] = np.log(diploids['YL2-A'])
diploids['Log-Gem'] = np.log(diploids['BL1-A'])

plt.figure()
# Set Cdt threshold
th = 8.5
plt.hist(diploids['Log-Cdt'],100);plt.xlabel('Log-Cdt1')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
diploids['High_Cdt'] = True
diploids.loc[diploids['Log-Cdt'] < th,'High_Cdt'] = False

# Set Cdt threshold
th = 10
plt.figure()
plt.hist(diploids['Log-Gem'],100);plt.xlabel('Log-Gem')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
diploids['High_Gem'] = True
diploids.loc[diploids['Log-Gem'] < th,'High_Gem'] = False

#%%
plt.figure()
sb.lmplot(data=diploids,x='Log-Cdt',y='Log-Gem',fit_reg=False,
          hue='High_Cdt',col='High_Gem', scatter_kws={'alpha':.01})

#%%

diploids.to_csv(path.join(dirname,'diploids.csv'))

diploids.loc[~diploids['High_Cdt'] & ~diploids['High_Gem'], 'Phase'] = 'G1'
diploids.loc[ diploids['High_Cdt'] & ~diploids['High_Gem'], 'Phase'] = 'G1'
diploids.loc[ diploids['High_Cdt'] & diploids['High_Gem'], 'Phase'] = 'S or G2'
diploids.loc[ ~diploids['High_Cdt'] & diploids['High_Gem'], 'Phase'] = 'S or G2'

(_,g1),(_,sg2) = diploids.groupby('Phase')

#%% Estimate CV + conf interv

Nboot = 1000
CV = pd.DataFrame(index=['G1','G1/S','S or G2'])

for phase,_df in diploids.groupby('Phase'):
    
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
plt.subplot(1,2,1)
sb.barplot(diploids,y='FSC-A',x='Phase'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           ,order=['G1','S or G2'])
plt.ylabel('CV of FSC')
plt.ylim([0,.25])
plt.title('Primary hepatocytes, cell cycle determined by FUCCI')

#%% Gate cell cycle based on DAPI only

# Set Cdt threshold
th = 0.41e6
plt.hist(diploids['VL1-A'],100);plt.xlabel('VL1-A')
plt.vlines(x=th,ymin=0,ymax=1000,color='r')
diploids['High_DAPI'] = True
diploids.loc[diploids['VL1-A'] < th,'High_DAPI'] = False

(_,twoN),(_,fourN) = diploids.groupby('High_DAPI')

#%% Plot the CVs as errorbars

# plt.subplot(1,2,2)
plt.figure()
sb.barplot(diploids,y='FSC-A',x='High_DAPI'
           ,estimator=stats.variation,errorbar=(lambda x: cvariation_ci_bootstrap(x,Nboot))
           )
plt.ylabel('CV of FSC')
plt.ylim([0,.25])
plt.title('Primary hepatocytes, cell cycle determined by DAPI')



