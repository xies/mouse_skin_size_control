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

# Load all the zebrafish
dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/CV from snapshot/zebrafish_ditalia/osx_fucci_26hpp_11_4_17/'
fish = pd.read_csv(path.join(dirname,'cell_size_by_cellcycle_position.csv'),index_col=0)

# Load models
dirname = '/Users/xies/OneDrive - Stanford/In vitro/CV from snapshot/CV model/G1timer_SG2sizer_asym05_grfluct05/'
model = pd.read_csv(path.join(dirname,'model_summary.csv'),index_col=0)

#%% Skin

bCV = cvariation_bootstrap(skin['Birth volume'],1000,subsample=100)
g1CV = cvariation_bootstrap(skin['G1 volume'],1000,subsample=100)
dCV = cvariation_bootstrap(skin['Division volume'],1000,subsample=100)

cv_skin = pd.DataFrame()
cv_skin.loc[:,'phase'] = ['Birth','G1S','Division']
cv_skin.loc[:,'CV'] = [bCV[0],g1CV[0],dCV[0]]
cv_skin.loc[:,'LB'] = [bCV[1],g1CV[1],dCV[1]]
cv_skin.loc[:,'UB'] = [bCV[2],g1CV[2],dCV[2]]
cv_skin['organism'] = 'skin'

#%% fish

bCV = cvariation_bootstrap(fish[fish['Phase'] == 'Birth']['Volume'],1000,subsample=100)
g1CV = cvariation_bootstrap(fish[fish['Phase'] == 'G1S']['Volume'],1000,subsample=100)
dCV = cvariation_bootstrap(fish[fish['Phase'] == 'Division']['Volume'],1000,subsample=100)

cv_fish = pd.DataFrame()
cv_fish.loc[:,'phase'] = ['Birth','G1S','Division']
cv_fish.loc[:,'CV'] = [bCV[0],g1CV[0],dCV[0]]
cv_fish.loc[:,'LB'] = [bCV[1],g1CV[1],dCV[1]]
cv_fish.loc[:,'UB'] = [bCV[2],g1CV[2],dCV[2]]
cv_fish['organism'] = 'fish'

#%% model

cv_model = pd.DataFrame()
cv_model.loc[:,'phase'] = ['Birth','G1S','Division']
cv_model.loc[:,'CV'] = []
cv_model.loc[:,'LB'] = [bCV[1],g1CV[1],dCV[1]]
cv_model.loc[:,'UB'] = [bCV[2],g1CV[2],dCV[2]]
cv_model['organism'] = 'fish'
