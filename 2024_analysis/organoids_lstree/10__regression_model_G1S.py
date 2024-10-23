#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 13:58:01 2024

@author: xies
"""

import numpy as np
import pandas as pd
from os import path
import seaborn as sb
from tqdm import tqdm

dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 5_2um/'
df5 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
df5['organoidID'] = 5
df5 = df5[ (df5['cellID'] !=77) | (df5['cellID'] != 120)]
# dirname = '/Users/xies/Library/CloudStorage/OneDrive-Stanford/In vitro/mIOs/organoids_LSTree/Position 2_2um/'
# df2 = pd.read_csv(path.join(dirname,'manual_cellcycle_annotations/cell_organoid_features_dynamic.csv'),index_col=0)
# df2['organoidID'] = 2
# df2 = df2[ (df2['cellID'] !=53) | (df2['cellID'] != 6)]

# df = pd.concat((df5,df2),ignore_index=True)
df = df5
df['organoidID_trackID'] = df['organoidID'].astype(str) + '_' + df['trackID'].astype(str)

# Derive some ratios
df['SA to vol ratio'] = df['Surface area'] / df['Nuclear volume']

#%% Preprocess
tracks = {trackID:t for trackID,t in df.groupby('organoidID_trackID')}

# First, drop everything but first G1/S frame
g1s_tracks = {}
for trackID,track in tracks.items():
    I = track['Auto phase']
    if I.sum() > 0:
        first_g1s_idx = np.where(I)[0][0]
        g1s_tracks[trackID] = track.iloc[0:first_g1s_idx+1]
        
g1s = pd.concat(g1s_tracks, ignore_index=True)
g1s = g1s.dropna()

#%% 

from sklearn.preprocessing import scale

# @todo: neighbor interiority, orientation, neighbor orientation
y = g1s['Auto phase']

feature_list = {'Nuclear volume':'nuc_vol',
                'Axial moment':'axial_moment',
                'Axial angle':'axial_angle',
                'Planar moment 1':'plan_moment_1',
                'Planar moment 2':'plan_moment_2',
                'Planar angle':'plan_angle',
                'Z':'z','Y':'y','X':'x',
                'SA to vol ratio':'SA_vol_ratio',
                'Mean Cdt1 intensity':'cdt1_int',
                'Mean Gem intensity':'gem_int',
                'Growth rates (sm)':'gr_sm',
                'Organoid interiority':'interior',
                'Mean curvature':'curvature',
                'Mean neighbor volume':'mean_neighb_vol',
                'Std neighbor volume':'std_neighb_vol',
                'Mean neighbor Cdt1 intensity':'mean_neighb_cdt1',
                'Mean neighbor Gem intensity':'mean_neighb_gem',
                'Local cell density':'local_density',
                'Change in local cell density':'delta_local_density',
                'Change in mean neighbor Cdt1':'delta_neighb_cdt1',
                'Change in mean neighbor Geminin':'delta_neighb_gem',
                'Change in Cdt1':'delta_cdt1',
                'Change in Geminin':'delta_gem',
                'Change in mean neighbor size':'delta_neighb_vol',
                'Change in std neighbor size':'delta_std_neighb_vol',
                # 'Orientation':'orientation',
                }

df_g1s = g1s.loc[:,feature_list.keys()]

df_g1s = df_g1s.rename(columns=feature_list)

#%%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

Niter = 25
X = scale(df_g1s)

score = np.zeros(Niter);
scores = pd.DataFrame()

for i in tqdm(range(Niter)):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = i)
    forest = RandomForestClassifier()
    
    forest.fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    
    scores.loc[i,'AP'] = average_precision_score(y_test,y_pred)
    scores.loc[i,'AUC'] = roc_auc_score(y_test,y_pred)
    
scores.plot.hist()

