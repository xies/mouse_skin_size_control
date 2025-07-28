#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:01:59 2025

@author: xies
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb

# General utils
from tqdm import tqdm
from os import path

dx = 0.25
dz = 1

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.csv'),
                     index_col=['Frame','TrackID'])

all_df = all_df[ ~all_df['Border']]
all_df = all_df[all_df['Keep until first differentiation'] == True]
tracks = [t for _,t in all_df.reset_index().groupby('TrackID')]

#%%

_X = all_df.drop(columns=all_df.columns[all_df.columns.str.startswith('cyto')])
_X = _X.drop(columns=['Right','Left','Mother','Daughter a','Daughter b'
                      ,'Time to differentiation'
                      ,'Apical area','Basal area','Basal orientation','Basal eccentricity'])
# _X = _X.loc[_X.isna().sum(axis=1) < 100]

# cols = _X.columns
# _X = _X.dropna()


# all_df.loc[all_df.isna().sum(axis=1) < 100]

_X
#%%

from sklearn.preprocessing import scale

feature2predict = 'Will differentiate'
y = _X[feature2predict]
X = scale(_X.drop(columns=feature2predict).values)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.2)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)



