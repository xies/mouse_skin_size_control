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

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/HMECs/08-21-2024 HMECs phosRB Hoechst mCherry-Gem'

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

#%%

filelist = glob(path.join(dirname,'*.*'))
conditions = ['WT','100nM']

datasets = []
for i,f in enumerate(filelist):
    _df = FCMeasurement(ID='Test Sample', datafile=f).data
    _df['Condition'] = conditions[i]

    #geminin should be on log scale
    _df['Log-geminin'] = np.log(_df['mCherry-A'])
    _df['Log-pRB'] = np.log(_df['BL1-A'])
    
    # put in the default gate values
    _df['Diploids'] = False
    _df['Phase'] = 'NA'
    _df['phosRB'] = 'Low'
    datasets.append(_df)

#%% Gate on singlets

gates2draw = {'singlets':['Hoechst-A','Hoechst-W'],
              'G1':['Hoechst-A','mCherry-A'],
              'SG2':['Hoechst-A','mCherry-A'],
              'phosRB':['Log-geminin','Log-pRB']}

singlet_selectors = []
for df in datasets:
    singlet_selectors.append(draw_gate(df,gating_axes=gates2draw['singlets']))

#%% Propagate gates and filter only diploids

for i in range(len(datasets)):
    datasets[i] = gate_data(datasets[i], singlet_selectors[i], gates2draw['singlets'],'Diploids',True)
    datasets[i] = datasets[i][datasets[i]['Diploids']]
    
#%% Gate on G1 and SG2

g1_selectors = []
for df in datasets:
    g1_selectors.append(draw_gate(df,gating_axes=gates2draw['G1'],alpha=0.01,title='G1'));

g2_selectors = []
for df in datasets:
    g2_selectors.append(draw_gate(df,gating_axes=gates2draw['SG2'],alpha=0.01,title='SG2'));

#%% Propagate gates

for i in range(len(datasets)):
    datasets[i] = gate_data(datasets[i], g1_selectors[i], gates2draw['G1'],'Phase','G1')
    datasets[i] = gate_data(datasets[i], g2_selectors[i], gates2draw['SG2'],'Phase','SG2')

#%% Gate on phosRB

rb_selectors = []
for df in datasets:
    rb_selectors.append(draw_gate(df,gating_axes=gates2draw['phosRB'],alpha=0.002,title='phosRB'));

#%% Propagate gates

for i in range(len(datasets)):
    datasets[i] = gate_data(datasets[i], rb_selectors[i], gates2draw['phosRB'],'phosRB','High')

#%% Concatenate all dataframes

df = pd.concat(datasets,ignore_index=True)

#%%

(_,wt),(_,palbo) = df.groupby('Condition')

Nboot = 1000
CV = pd.DataFrame()
CV_ci = pd.DataFrame()

_df = wt[(wt['Phase'] == 'G1') & (wt['phosRB'] == 'Low')]
CV.loc['WT','pre-G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['WT','pre-G1S'] = (ub-lb)/2

_df = wt[(wt['Phase'] == 'G1') & (wt['phosRB'] == 'High')]
CV.loc['WT','G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['WT','G1S'] = (ub-lb)/2

_df = palbo[(palbo['Phase'] == 'G1') & (palbo['phosRB'] == 'Low')]
CV.loc['palbo','pre-G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['palbo','pre-G1S'] = (ub-lb)/2

_df = palbo[(palbo['Phase'] == 'G1') & (palbo['phosRB'] == 'High')]
CV.loc['palbo','G1S'],lb,ub = cvariation_bootstrap(_df['SSC-A'].values,1000)
CV_ci.loc['palbo','G1S'] = (ub-lb)/2

print(CV)

plt.errorbar([1,2],CV.loc['WT'].values,yerr=CV_ci.loc['WT'].values)
plt.errorbar([1,2],CV.loc['palbo'].values,yerr=CV_ci.loc['palbo'].values)

plt.gca().set_xticks([1,2]);plt.gca().set_xticklabels(['pre-G1S','G1S'])
plt.xlabel('Cell cycle phase')
plt.ylabel('CV in side scatter')
plt.legend(['Normal','100nM Palbo 48h'])

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


