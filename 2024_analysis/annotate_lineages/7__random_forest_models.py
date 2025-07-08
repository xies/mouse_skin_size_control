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

def get_balanced_df_by_category(df,category):
    # assert(logical.dtype == bool)
    
    # Find the category with least members
    categories = {cat:mem for cat, mem in df.groupby(category)}
    num_per_category = np.array([len(mem) for mem in categories.values()])
    smallest_category = list(categories.keys())[num_per_category.argmin()]
    
    output = []
    for cat, member in categories.items():
        if cat == smallest_category:
            output.append(member)
        else:
            output.append(member.sample(num_per_category.min()))
    output = pd.concat(output,ignore_index=True)

    return output

#%% Categorize cell from birth frame

Niter = 100

df = all_df[all_df['Fate known','Meta']]
df = df[ ~df['Border','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

births = df[df['Birth frame','Meta']]
births[('Relative basal area','Measurement')] = \
    births['Basal area','Measurement'] / births['Mean adjac Basal area','Measurement']
features2drop = [f for f in births.columns.get_level_values(0) if 'Time to differentiation' in f]
# Censor height information
features2drop = features2drop + ['Z','Mean curvature - cell coords','Z-cyto','Height to BM',
                                 'Basal area','Apical area']
# Censor all exponential rates (but not mother's)
features2drop = features2drop + [f for f in births.columns.get_level_values(0)
                                 if ('exponential' in f \
                                     and 'frame prior' not in f \
                                         and 'adjac' not in f)]
births = births.drop(columns=features2drop)

metas = pd.DataFrame()
metas['name'] = births.xs('Meta',level=1,axis=1).columns
features = pd.DataFrame()

measurements = births.xs('Measurement',level=1,axis=1)
features['name'] = measurements.columns
features =features.set_index('name')
features['Num NA'] = measurements.isna().sum(axis=0)
# features = features.drop(features.loc[features.index.str.startswith('cyto_')].index)

feature_names = features.index
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_ = get_balanced_df_by_category(births, births['Will differentiate','Meta'].values)
    
    y = df_['Will differentiate','Meta']
    X = preprocessing.scale(df_.xs('Measurement',level=1,axis=1))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)

    importances.loc[i] = forest.feature_importances_

plt.title('Predict self-fate from birth frame (random forest)')
sb.heatmap(avg_confusion.mean(axis=0),annot=True,vmin=0,vmax=0.5)
plt.gca().set_xticks([0.5,1.5],labels=['Divide','Differentiate'])
plt.gca().set_yticks([0.5,1.5],labels=['Divide','Differentiate'])
plt.xlabel('Measured fate')
plt.ylabel('Predicted fate')

plt.figure()
plt.title('Predict self-fate from birth frame (random forest)')
print(importances.mean().sort_values().tail(20))
importances.mean().sort_values().tail(20).plot.bar(); plt.tight_layout();
plt.ylabel('Feature importance')

#%% Categorize cell from mother division frame

BALANCE = True
Niter = 100

# feature2predict = 'At least one daughter differentiated'
# feature2predict = 'Both daughters differentiated'
feature2predict = 'Num daughter differentiated'

df = all_df[all_df['Fate known','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

divisions = df[df[('Divide next frame','Meta')]]
divisions = divisions[~divisions['Border','Meta']]
divisions = divisions.reset_index().set_index('TrackID')
divisions[('Relative basal area','Measurement')] = \
    divisions['Basal area','Measurement'] / divisions['Mean adjac Basal area','Measurement']

features2drop = [f for f in df.columns.get_level_values(0) if 'Time' in f]
features2drop += [f for f in df.columns.get_level_values(0) if 'Age' in f]

features2drop = [(f,'Measurement') for f in features2drop]
features2drop = features2drop + [('Keep until first differentiation','Meta')]
divisions = divisions.drop(columns=features2drop).dropna(
    subset=[(feature2predict,'Meta')])

feature_names = divisions.xs('Measurement',axis=1,level=1).columns
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,3,3))

for i in tqdm(range(Niter)):
    
    if BALANCE:
        df_ = get_balanced_df_by_category(divisions,
                                          divisions[(feature2predict,'Meta')].values)
    else:
        df_ = divisions
        
    _tmp = df_.xs('Measurement',level=1,axis=1).copy()
    _tmp[feature2predict] = df_[(feature2predict,'Meta')].copy()
    
    y = _tmp[feature2predict].values
    X = preprocessing.scale(_tmp.drop(columns=feature2predict))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.1)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)
    

    importances.loc[i] = forest.feature_importances_

sb.heatmap(avg_confusion.mean(axis=0),annot=True,vmin=0,vmax=0.5)
plt.xlabel('Predicted # of differentiated daughter')
plt.ylabel('Measured #')
print(importances.mean().sort_values().tail(20))
importances.mean().sort_values().tail(20).plot.bar(); plt.tight_layout();
plt.ylabel('Feature importance')

#%%




#%% Categorize cell from mother division frame -1

from measurements import get_prev_or_next_frame

BALANCE = True
Niter = 100

# feature2predict = 'At least one daughter differentiated'
# feature2predict = 'Both daughters differentiated'
feature2predict = 'Num daughter differentiated'

df = all_df[all_df['Fate known','Meta']]
df = df[ df['Cell type','Meta'] == 'Basal']

divisions = df[df[('Divide next frame','Meta')]]
divisions = divisions[~divisions['Border','Meta']]
divisions = divisions.reset_index().set_index('TrackID')

prev_div_frame = [get_prev_or_next_frame(all_df,f,direction='prev') 
                  for _,f in divisions.iterrows()]

prev_div_frame = pd.concat(prev_div_frame,axis=1).T
prev_div_frame = df[df[('Divide next frame','Meta')]]
prev_div_frame = prev_div_frame[~prev_div_frame['Border','Meta']]
prev_div_frame = prev_div_frame.reset_index().set_index('TrackID')


features2drop = [f for f in df.columns.get_level_values(0) if 'Time' in f]
features2drop += [f for f in df.columns.get_level_values(0) if 'Age' in f]

features2drop = [(f,'Measurement') for f in features2drop]
features2drop = features2drop + [('Keep until first differentiation','Meta')]
divisions = divisions.drop(columns=features2drop).dropna(
    subset=[(feature2predict,'Meta')])

feature_names = divisions.xs('Measurement',axis=1,level=1).columns
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,3,3))

for i in tqdm(range(Niter)):
    
    if BALANCE:
        df_ = get_balanced_df_by_category(prev_div_frame,
                                          prev_div_frame[(feature2predict,'Meta')].values)
    else:
        df_ = divisions
        
    _tmp = df_.xs('Measurement',level=1,axis=1).copy()
    _tmp[feature2predict] = df_[(feature2predict,'Meta')].copy()
    
    y = _tmp[feature2predict].values
    X = preprocessing.scale(_tmp.drop(columns=feature2predict))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.1)
    
    forest = ensemble.RandomForestRegressor().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)
    

    importances.loc[i] = forest.feature_importances_

sb.heatmap(avg_confusion.mean(axis=0),annot=True,vmin=0,vmax=0.5)
plt.xlabel('Predicted # of differentiated daughter')
plt.ylabel('Measured #')
print(importances.mean().sort_values().tail(20))


#%%%

sb.catplot(divisions.droplevel(level=1,axis=1),x='Num daughter differentiated',y='Relative basal area',kind='violin')
sb.catplot(divisions.droplevel(level=1,axis=1),x='Num daughter differentiated',y='Relative basal area',kind='violin')



