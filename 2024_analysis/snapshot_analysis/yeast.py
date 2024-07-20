#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:32:09 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import seaborn as sb
from basicUtils import nonans

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/Jacob_yeast/'

filelist = ['MS358_expt1_G1_2020-10-06_dataset_aligned_volume.csv'
      ,'MS358_expt2_G1_2021-03-08_dataset_aligned_volume.csv'
      ,'MS358_expt3_G1_2021-03-29_dataset_aligned_volume.csv'
      ,'MS358_expt1_SG2M_2020-10-06_dataset_aligned_volume.csv'
      ,'MS358_expt2_SG2M_2021-03-08_dataset_aligned_volume.csv'
      ,'MS358_expt3_SG2M_2021-03-29_dataset_aligned_volume.csv']

phases = ['G1','G1','G1','SG2','SG2','SG2']
exp = [1,2,3,1,2,3]
dsnames = ['G1_1','G1_2','G1_3','SG2_1','SG2_2','SG2_3']
# dsnames = list(zip(phases,exp))

#%%

datasets = {}

for i,f in enumerate(filelist):
    # df_ = pd.read_csv(path.join(dirname,f),column=range())
    M = np.genfromtxt(path.join(dirname,f), delimiter=',')
    Ncells,Nframes = M.shape
    df_ = pd.DataFrame(M,index=range(Ncells),columns=range(Nframes))
    datasets[dsnames[i]] = df_
   
Mg1 = datasets[('G1',1)]
CV_g1_1 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)
Msg2 = datasets[('G1',2)]
CV_g1_2 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)
Mg1 = datasets[('G1',3)]
CV_g1_3 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)
Msg2 = datasets[('SG2',1)]
CV_g2_1 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)
M = datasets[('SG2',2)]
CV_g2_2 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)
M = datasets[('SG2',3)]
CV_g2_3 = np.nanstd(M,axis=1)/np.nanmean(M,axis=1)

#%% Interp

colors = {'G1':'b','SG2':'r'}
tnew = np.linspace(0,1,10)
CVs = pd.DataFrame(index = dsnames,columns= tnew)
interp_growth_curves = {}

for name,M in datasets.items():
    
    Minterp = np.zeros((M.shape[0],10))
    for cellID,trace in M.iterrows():
        
        I = ~np.isnan(trace)
        y = trace[I]
        t = np.linspace(0,1,len(y))
        
        Minterp[cellID,:] = np.interp(tnew,t,y)
    
    interp_growth_curves[name] = Minterp
    CVs.loc[name] = np.std(Minterp,axis=0) / np.mean(Minterp,axis=0)
# CVs = CVs.reset_index()

sb.relplot(CVs.T,kind='line')

Mg1 = np.vstack((interp_growth_curves['G1_1'],interp_growth_curves['G1_2'],interp_growth_curves['G1_3']))
Mgsg2 = np.vstack((interp_growth_curves['SG2_1'],interp_growth_curves['SG2_2'],interp_growth_curves['SG2_3']))

G1_CV = np.std(Mg1,axis=0) / np.mean(Mg1,axis=0)
SG2_CV = np.std(Mgsg2,axis=0) / np.mean(Mgsg2,axis=0)

#%%


