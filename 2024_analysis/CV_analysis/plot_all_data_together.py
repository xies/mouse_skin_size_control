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

Pb_v_g1 = cv_difference_pvalue(skin['Birth volume'],skin['G1 volume'],Nboot=1000)
Pdiv_v_g1 = cv_difference_pvalue(skin['Division volume'],skin['G1 volume'],Nboot=1000)
Pb_v_div = cv_difference_pvalue(skin['Birth volume'],skin['Division volume'],Nboot=1000)
print('--- Skin CV bootstrap tests --- ')
print(f'Birth v. G1: P = {Pb_v_g1}')
print(f'Division v. G1: P = {Pdiv_v_g1}')
print(f'Birth v. division: P = {Pb_v_div}')

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

Pb_v_g1 = cv_difference_pvalue(fish[fish['Phase'] == 'Birth']['Volume'],
                               fish[fish['Phase'] == 'G1S']['Volume'],Nboot=1000)
Pdiv_v_g1 = cv_difference_pvalue(fish[fish['Phase'] == 'Division']['Volume'],
                                 fish[fish['Phase'] == 'G1S']['Volume'],Nboot=1000)
Pb_v_div = cv_difference_pvalue(fish[fish['Phase'] == 'Division']['Volume'],
                                 fish[fish['Phase'] == 'Birth']['Volume'],Nboot=1000)
print('--- Fish CV bootstrap tests --- ')
print(f'Birth v. G1: P = {Pb_v_g1}')
print(f'Division v. G1: P = {Pdiv_v_g1}')
print(f'Birth v. division: P = {Pb_v_div}')

#%% Model

# Load models
dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/G1timer_SG2sizer_asym05_grfluct05/'
model = pd.read_csv(path.join(dirname,'model_summary.csv'),index_col=0)

fig,ax1 = plt.subplots()
err = model.loc['sizer550_timer14',['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc['sizer550_timer14',['Birth CV','G1S CV','Div CV']].values
ax1.errorbar([1-0.05,2-0.05,3-0.05],model.loc['sizer550_timer14',['Birth CV','G1S CV','Div CV']],err)
err = model.loc['timer40_sizer',['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc['timer40_sizer',['Birth CV','G1S CV','Div CV']].values
ax1.errorbar([1,2,3],model.loc['timer40_sizer',['Birth CV','G1S CV','Div CV']],err)
ax1.tick_params(axis='y', color='C0', labelcolor='C0')
plt.ylabel('Change in C.V. of cell size')

ax2 = ax1.twinx()
err = model.loc['adder_100_adder_300',['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc['adder_100_adder_300',['Birth CV','G1S CV','Div CV']].values
ax2.errorbar([1.05,2.05,3.05],model.loc['adder_100_adder_300',['Birth CV','G1S CV','Div CV']],err,color='r')
ax2.tick_params(axis='y', color='C0', labelcolor='r')
plt.legend(['G1/S adder; S/G2/M adder'])
plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])

#%% Multiplot all model configs

colors = {'sizer':'C0','adder':'r','timer':'C1'}

for model_name in model.index:
    
    err = model.loc[model_name,['Birth CV UB','G1S CV UB','Div CV UB']].values - model.loc[model_name,['Birth CV','G1S CV','Div CV']].values
    plt.errorbar([1,2,3],model.loc[model_name,['Birth CV','G1S CV','Div CV']],err
                 , color=colors[model.loc[model_name,'G1 model']])
    plt.xticks([1,2,3],labels=['Birth','G1/S','Division'])



