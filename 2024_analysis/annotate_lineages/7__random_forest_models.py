#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 15:07:15 2025

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

from sklearn import preprocessing, model_selection
from sklearn import ensemble, metrics

dirname = '/Users/xies/OneDrive - Stanford/Skin/Mesa et al/W-R1/'
all_df = pd.read_pickle(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback_history.pkl'))
all_df = all_df.drop_duplicates().sort_index()
all_tracks = {trackID:t for trackID,t in all_df.reset_index().groupby('TrackID')}

#%%

def get_balanced_df_by_category(df,logical):
    assert(logical.dtype == bool)
    
    trues = df[logical]
    falses = df[~logical]
    if len(trues) > len(falses):
        output = pd.concat((df[~logical], df[logical].sample(len(falses))),ignore_index=True)
    elif len(trues) < len(falses):
        output = pd.concat((df[logical], df[~logical].sample(len(trues))),ignore_index=True)
    else:
        output = df
        
    return output

#%% Filter + separate the bookkeeping columns

df = all_df[all_df['Fate known','Meta']]
df = df[ ~df['Border','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

features2drop = [f for f in df.columns.get_level_values(0) if 'Time to differentiation' in f]
features2drop = features2drop + ['Z','Mean curvature - cell coords','Z-cyto','Height to BM']
features2drop = [(f,'Measurement') for f in features2drop]
df = df.drop(columns=features2drop)

tracks = {trackID:t for trackID,t in df.reset_index().groupby('TrackID') if not np.any(t['Border','Meta'])}

metas = pd.DataFrame()
metas['name'] = df.xs('Meta',level=1,axis=1).columns

features = pd.DataFrame()

measurements = df.xs('Measurement',level=1,axis=1)
features['name'] = measurements.columns

features =features.set_index('name')
features['Num NA'] = measurements.isna().sum(axis=0)
# features = features.drop(features.loc[features.index.str.startswith('cyto_')].index)

#% Categorize cell from birth frame

Niter = 100
birth = df[df['Birth frame','Meta']]

feature_names = features.index
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_ = get_balanced_df_by_category(birth, birth['Will differentiate','Meta'].values)
    
    y = df_['Will differentiate','Meta']
    X = preprocessing.scale(df_.xs('Measurement',level=1,axis=1))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)

    importances.loc[i] = forest.feature_importances_

sb.heatmap(avg_confusion.mean(axis=0),annot=True)
print(importances.mean().sort_values().tail(20))

#%% Categorize cell from mother division frame

Niter = 25

df = all_df[all_df['Fate known','Meta']]
df = df[ ~df['Border','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

divisions = df[df[('Divide next frame','Meta')]]
divisions

features2drop = [f for f in df.columns.get_level_values(0) if 'Time' in f]
features2drop += [f for f in df.columns.get_level_values(0) if 'Age' in f]

features2drop = [(f,'Measurement') for f in features2drop]
features2drop = features2drop + [('Keep until first differentiation','Meta')]
divisions = divisions.drop(columns=features2drop).dropna(subset=[('At least one differentiated daughter','Meta')])

feature_names = divisions.xs('Measurement',axis=1,level=1).columns
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_ = get_balanced_df_by_category(divisions, divisions['At least one differentiated daughter','Meta'].astype(bool).values)
    _tmp = df_.xs('Measurement',level=1,axis=1).copy()
    _tmp['At least one differentiated daughter'] = df_['At least one differentiated daughter','Meta'].copy()
    
    y = _tmp['At least one differentiated daughter'] > 0
    X = preprocessing.scale(_tmp.drop(columns='At least one differentiated daughter'))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)

    importances.loc[i] = forest.feature_importances_

sb.heatmap(avg_confusion.mean(axis=0),annot=True,vmin=0,vmax=0.5)
print(importances.mean().sort_values().tail(20))



