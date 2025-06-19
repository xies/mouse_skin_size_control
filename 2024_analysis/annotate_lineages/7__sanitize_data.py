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
all_df = pd.read_csv(path.join(dirname,'Mastodon/single_timepoints_dynamics_aggregated_lookback.csv'),
                     index_col=['Frame','TrackID'])
all_df = all_df[all_df['Fate known']]

all_df = all_df[ ~all_df['Border']]
all_df = all_df[ all_df['Cell type'] == 'Basal']
tracks = [t for _,t in all_df.reset_index().groupby('TrackID') if np.all(~t.Border)]

features = pd.DataFrame()
features['name'] = all_df.columns.drop(['Cell type','Mother','Daughter a','Daughter b','Sister','Left','Right',
                                        'Time to differentiation','Keep until first differentiation',
                                        # 'Will differentiate',
                                        'Will divide','Z','Z-cyto','Complete cycle',
                                        'Cell volume smoothed growth rate',
                                        'Cell volume exponential growth rate',
                                        'Mean FUCCI intensity smoothed growth rate',
                                        'Mean FUCCI intensity smoothed',
                                        'Total H2B intensity smoothed growth rate',
                                        'Cutoff',
                                        'Delaminate next frame',
                                        'Time',
                                        ])
features =features.set_index('name')
features['Num NA'] = all_df.isna().sum(axis=0)
features = features.drop(features.loc[features.index.str.startswith('cyto_')].index)

all_df = all_df[features.index].dropna()

#%% Categorize cell from birth frame

tracks = [t for _,t in all_df.reset_index().groupby('TrackID') if np.all(~t.Border)]

tracks = {trackID:t for trackID,t in all_df.groupby('TrackID') if t.iloc[0]['Born']}

birth = pd.concat([t.iloc[0] for t in tracks.values()],ignore_index=True,axis=1).T
birth.index = list(tracks.keys())

I_true = all_df['Will differentiate']
I_false = ~all_df['Will differentiate']

feature_names = birth.drop(columns='Will differentiate').columns
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((100,2,2))
for i in tqdm(range(100)):
    df_ = all_df.loc[I_true]
    df_ = pd.concat((df_,all_df.loc[I_false].sample(len(df_))))
    
    y = df_['Will differentiate']
    X = preprocessing.scale(df_.drop(columns='Will differentiate'))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)

    importances.loc[i] = forest.feature_importances_

print(importances.mean().sort_values().tail(20))

#%%

sb.heatmap(avg_confusion.mean(axis=0),annot=True)

sorted_features = importances.mean().sort_values().tail(20).index
plt.figure()
sb.barplot(importances[sorted_features].melt(),x='variable',y='value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.xlabel('Feature @ birth')
plt.ylabel('Feature importance')

#%% Predict next time frame

I_true = all_df['Delaminate next frame']
I_false = ~all_df['Delaminate next frame']

feature_names = all_df.drop(columns='Delaminate next frame').columns
importances = pd.DataFrame(columns=feature_names)
for i in tqdm(range(100)):
    df_ = all_df.loc[I_true]
    df_ = pd.concat((df_,all_df.loc[I_false].sample(len(df_))))
    
    y = df_['Delaminate next frame']
    X = preprocessing.scale(df_.drop(columns='Delaminate next frame'))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    
    C = metrics.confusion_matrix(y_test,y_pred)

    importances.loc[i] = forest.feature_importances_
    
print(importances.mean().sort_values().tail(20))


