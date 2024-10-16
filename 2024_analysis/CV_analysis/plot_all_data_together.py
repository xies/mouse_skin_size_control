#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:44:12 2024

@author: xies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mathUtils import cvariation_bootstrap, cv_difference_pvalue
from os import path
from scipy import stats
import pickle as pkl
from basicUtils import nonans

#%% Load empirical data

# Load data from skin
r1 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/2020 CB analysis/exports/dataframe.pkl')
r2 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R2/2020 CB analysis/exports/dataframe.pkl')
r5 = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5/tracked_cells/dataframe.pkl')
r5f = pd.read_pickle('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R5-full/tracked_cells/dataframe.pkl')
skin = pd.concat((r1,r2,r5,r5f))

bCV = cvariation_bootstrap(skin['Birth volume'],1000)
g1CV = cvariation_bootstrap(skin['G1 volume'],1000)
dCV = cvariation_bootstrap(skin['Division volume'],1000)

cv_skin = pd.DataFrame(index=['Birth','G1S','Division'])
# cv_skin.loc.index = ['Birth','G1S','Division']
cv_skin.loc[:,'CV'] = [bCV[0],g1CV[0],dCV[0]]
cv_skin.loc[:,'LB'] = [bCV[1],g1CV[1],dCV[1]]
cv_skin.loc[:,'UB'] = [bCV[2],g1CV[2],dCV[2]]
cv_skin['organism'] = 'skin'
cv_skin.loc[:,'Err'] = (cv_skin['UB'] - cv_skin['CV'])

plt.figure()
plt.errorbar([1,2,3],cv_skin.loc[['Birth','G1S','Division']]['CV'], cv_skin.loc[['Birth','G1S','Division']]['Err'])
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])
plt.ylabel('C.V. in cell volume')
plt.title('Mouse basal layer cells')

#%% fish

# Load all the zebrafish
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'
fish = pd.read_csv(path.join(dirname,'cell_size_by_cellcycle_position.csv'),index_col=0)

bCV = cvariation_bootstrap(fish[fish['Phase'] == 'Birth']['Volume'],1000,subsample=100)
g1CV = cvariation_bootstrap(fish[fish['Phase'] == 'G1S']['Volume'],1000,subsample=100)
dCV = cvariation_bootstrap(fish[fish['Phase'] == 'Division']['Volume'],1000,subsample=100)

cv_fish = pd.DataFrame(index=['Birth','G1S','Division'])
cv_fish.loc[:,'CV'] = [bCV[0],g1CV[0],dCV[0]]
cv_fish.loc[:,'LB'] = [bCV[1],g1CV[1],dCV[1]]
cv_fish.loc[:,'UB'] = [bCV[2],g1CV[2],dCV[2]]
cv_fish['organism'] = 'fish'
cv_fish.loc[:,'Err'] = (cv_fish['UB'] - cv_fish['LB'])/2

plt.figure()
plt.errorbar([1,2,3],cv_fish.loc[['Birth','G1S','Division']]['CV'], cv_fish.loc[['Birth','G1S','Division']]['Err'])
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])
plt.ylabel('C.V. in cell volume')
plt.title('Zebrafish osteoblasts')

#%% yy plot

fig, ax1 = plt.subplots()

ax1.errorbar([1,2,3],cv_fish.loc[['Birth','G1S','Division']]['CV'], cv_fish.loc[['Birth','G1S','Division']]['Err'])
ax1.tick_params(axis='y', color='C0', labelcolor='C0')
plt.legend(['Mouse skin'])
plt.ylabel('C.V. in cell volume')

ax2 = ax1.twinx()
ax2.errorbar([1,2,3],cv_skin.loc[['Birth','G1S','Division']]['CV'], cv_skin.loc[['Birth','G1S','Division']]['Err'],
             color='r')
plt.legend(['Zebrafish'])
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])
ax2.tick_params(axis='y', color='r', labelcolor='r')
plt.ylabel('C.V. in nuclear volume')

plt.show()

#%% Model

# Load models
dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/G1timer_SG2sizer_asym05_grfluct05/'
model = pd.read_csv(path.join(dirname,'model_summary.csv'),index_col=0)

err = model.loc['sizer450_timer14',['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc['sizer450_timer14',['Birth CV','G1S CV','Div CV']].values
fig,ax1 = plt.subplots()
ax1.errorbar([1,2,3],model.loc['sizer450_timer14',['Birth CV','G1S CV','Div CV']],err)
ax1.tick_params(axis='y', color='C0', labelcolor='C0')
plt.legend(['Sizer model'])
plt.ylabel('C.V. in cell size')

ax2 = ax1.twinx()
err = model.loc['adder_100_adder_100',['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc['adder_100_adder_100',['Birth CV','G1S CV','Div CV']].values
ax2.errorbar([1.05,2.05,3.05],model.loc['adder_100_adder_100',['Birth CV','G1S CV','Div CV']],err,color='r')
ax2.tick_params(axis='y', color='C0', labelcolor='r')
plt.legend(['Adder model'])
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])



