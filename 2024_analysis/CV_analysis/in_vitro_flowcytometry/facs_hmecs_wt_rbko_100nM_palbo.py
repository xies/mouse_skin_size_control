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
# from sklearn import mixture
from glob import glob
from os import path

from SelectFromCollection import SelectFromCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from scipy import stats
from mathUtils import cvariation_bootstrap
from ifUtils import Gate

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Flow/HMECs/07-01-2024 HMECs WT RBKO palbo repeat'

#%% Load and preprocess variables

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

#%% Define the gates

fsc_ssc_gate = Gate('fsc-scc','FSC-A','SSC-A')
diploid_gate = Gate('diploids','Hoechst-H','Hoechst-W')
geminin_gate = Gate('Geminin_high','Hoechst-A','Log-geminin')

#%% gate on real cells

fsc_ssc_gate.draw_gates(df)

#%% Throwaway non cells

cells = df[fsc_ssc_gate.get_gated_indices(df)]
fsc_ssc_gate.draw_gate_as_patch(df)

#%% Gate on diploids

diploid_gate.draw_gates(cells,alpha=0.005)

#%% Throwaway non diploids

diploids = df[diploid_gate.get_gated_indices(df)]
diploid_gate.draw_gate_as_patch(cells)

#%% Gate on Geminin+

geminin_gate.draw_gates(diploids)

#%% Propagate geminin status

I = geminin_gate.get_gated_indices(diploids)
diploids['Geminin_high'] = False
diploids.loc[I,'Geminin_high'] = True

#%% Build CV table using bootstrap

Nboot = 1000
CV = []

for (condition,phase),_df in diploids.groupby(['Condition','Geminin_high']):
    
    # _cv,_lb,_ub = cvariation_bootstrap(_df['SSC-A'],Nboot=1000,subsample=10000)
    _cv,_lb,_ub = cvariation_bootstrap(_df['Alexa Fluor™ 647-A'],Nboot=1000,subsample=10000)

    CV.append(pd.DataFrame({'Condition':condition,
                         'Geminin_high':phase,
                         'CV':_cv,
                         'UB':_ub,
                         'LB':_lb,
                         },index=[0]))
CV = pd.concat(CV,ignore_index=True)

#%% Plot CV table

plt.figure()
for i,(condition,_df) in enumerate(CV.groupby('Condition')):
    
    if condition.startswith('WT'):
        plt.subplot(1,2,1)
        plt.title('Wild type')
    if condition.startswith('RBKO'):
        plt.subplot(1,2,2)
        plt.title('RB-KO')
    
    g1 = _df[~_df['Geminin_high']]
    sg2 = _df[_df['Geminin_high']]
    _cv = np.hstack((g1['CV'].values, sg2['CV'].values))
    _yerr = np.hstack(( g1['UB']-g1['LB'].values, sg2['UB']-sg2['LB'].values ))/2
    plt.errorbar([1,2],_cv,yerr=_yerr)
    plt.ylim([0,.5])
    plt.xticks([1,2],labels=['G1','Post-G1/S'])
    plt.xlabel('Cell cycle phase')
    plt.ylabel('CV in cell size (SSC-A)')

    print(condition)
plt.legend(['DMSO','Palbo'])

#%% Plot the CVs as errorbars

# def plot_size_CV_subplots(df,x,title=None):
    
#     plt.figure()
#     plt.subplot(1,3,1)
#     sb.barplot(df,y='FSC-A',x=x
#                ,estimator=stats.variation,errorbar=(lambda x: cvariation_bootstrap(x,Nboot)[:1:2])
#                ,order=[False,True])
#     plt.ylabel('CV of FSC')
    
#     plt.subplot(1,3,2)
#     sb.barplot(df,y='SSC-A',x=x
#                ,estimator=stats.variation,errorbar=(lambda x: cvariation_bootstrap(x,Nboot)[:1:2])
#                ,order=[False,True])
#     plt.ylabel('CV of SSC')
#     plt.title(title)
    
#     plt.subplot(1,3,3)
#     sb.barplot(df,y='Alexa Fluor™ 647-A',x=x
#                ,estimator=stats.variation,errorbar=(lambda x: cvariation_bootstrap(x,Nboot)[:1:2])
#                ,order=[False,True])
#     plt.ylabel('CV of SE-647')


# plot_size_CV_subplots(diploids[diploids['Condition'] == 'WT'],'Geminin_high','HMEC WT')
# plot_size_CV_subplots(diploids[diploids['Condition'] == 'WT palbo'],'Geminin_high','HMEC WT 100nM Palbo')

# plot_size_CV_subplots(diploids[diploids['Condition'] == 'RBKO'],'Geminin_high','HMEC RBKO')
# plot_size_CV_subplots(diploids[diploids['Condition'] == 'RBKO palbo'],'Geminin_high','HMEC RBKO 100nM Palbo')


