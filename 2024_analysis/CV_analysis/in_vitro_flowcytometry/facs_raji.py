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
from mathUtils import cvariation_ci, cvariation_bootstrap


def draw_gate(df,gating_axes,alpha=0.005,title='Gate'):
    plt.figure()
    pts = plt.scatter(df[gating_axes[0]],df[gating_axes[1]],alpha=alpha)
    plt.xlabel(gating_axes[0]); plt.ylabel(gating_axes[1])
    selector = SelectFromCollection(plt.gca(),pts)
    plt.title(title)
    return selector

def gate_data(df,selector,gating_axes,field_name,value):
    verts = np.array(selector.poly.verts)
    x = verts[:,0];y = verts[:,1]
    p_ = Path(np.array([x,y]).T)
    I = np.array([p_.contains_point([x,y]) for x,y in zip(df[gating_axes[0]],df[gating_axes[1]])])
    df.loc[I,field_name] = value
    return df

#%% Rep 1

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/06-23-2024 Raji Hoechst EdU'

#%%

filelist = glob(path.join(dirname,'*Raji*.fcs'))
df = FCMeasurement(ID='Test Sample', datafile=filelist[0]).data
df['Diploids'] = False
df['Phase'] = 'NA'
df['Log-EdU'] = np.log(df['RL1-A'])

#%% Gate on singlets

gates2draw = {'singlets':['VL1-A','VL1-W'],
              'G1':['VL1-A','Log-EdU'],
              'SG2':['VL1-A','Log-EdU']}

singlet_selectors = draw_gate(df,gating_axes=gates2draw['singlets'])

#%% Propagate gates and filter only diploids

df = gate_data(df, singlet_selectors, gates2draw['singlets'],'Diploids',True)
df = df[df['Diploids']]

#%% Gate on G1 and SG2

g1_selectors = draw_gate(df,gating_axes=gates2draw['G1'],alpha=0.01,title='G1')
s_selectors = draw_gate(df,gating_axes=gates2draw['SG2'],alpha=0.01,title='S')

#%% Propagate gates

df = gate_data(df, g1_selectors, gates2draw['G1'],'Phase','G1')
df = gate_data(df, s_selectors, gates2draw['SG2'],'Phase','S')

#%%

Nboot = 1000
CV = pd.DataFrame()
CV_ci = pd.DataFrame()

_df = df[df['Phase'] == 'G1']
CV.loc['Raji','pre-G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['WT','G1'] = (ub-lb)/2

_df = df[df['Phase'] == 'S']
CV.loc['Raji','G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['WT','S'] = (ub-lb)/2
print(CV)

plt.errorbar([1,2],CV.loc['Raji'].values,yerr=CV_ci.loc['WT'].values)

plt.gca().set_xticks([1,2]);plt.gca().set_xticklabels(['G1S','S'])
plt.xlabel('Cell cycle phase')
plt.ylabel('CV in side scatter')
plt.legend(['Normal','100nM Palbo 48h'])



