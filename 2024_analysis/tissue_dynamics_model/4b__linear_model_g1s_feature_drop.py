#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 18:18:35 2022

@author: xies
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from basicUtils import *
from os import path
from tqdm import tqdm

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 
from scipy.stats import stats

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]


X = df_g1s.drop(columns=['time_g1s'])
y = df_g1s['time_g1s']

#Add interaction effects ?
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

#%% Recursive feature drop with RF

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train,y_train)

param_names_in_order = X.columns[np.argsort(forest.feature_importances_)[::-1]]

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn import linear_model

for num_feat in np.arange(1,10):
    selector = RFE(RandomForestRegressor(n_estimators=100, random_state=i), n_features_to_select=num_feat)
    selector = selector.fit(X_train,y_train)
    print(f'Num feature = {num_feat}: {X.columns.values[selector.support_]}')

#%% Drop feature in order of importance and look at R2 (cross replicate?)
#NB: Something is weird... need to figure out

# X_sorted = X[param_names_in_order]

Rsq = np.zeros(23)
res = np.zeros(23)

features2drop = []

for i in range(23):

    X_dropped = X_sorted.drop(columns=features2drop)    
    X_train,X_test,y_train,y_test = train_test_split(X_dropped,y,test_size=0.2,random_state = 42)

    # Compute current model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train,y_train)
    largest_effect_param = X_train.columns[np.argmax(np.abs(model.feature_importances_))]
    
    features2drop.append(largest_effect_param)
    
    # Find the largest effect size predcitor
    ypred = model.predict(X_test)
    res[i] = corrcoef(y_test,ypred)[0,1]**2

plt.figure()
model_scores = pd.DataFrame(-np.diff(res),index=features2drop[:-1],columns=['Importance'])
model_scores['Name'] = model_scores.index
sb.barplot(data=model_scores,x='Name',y='Importance');

plt.figure()
model_scores = pd.DataFrame(res,index=features2drop,columns=['Importance'])
model_scores['Name'] = model_scores.index
sb.barplot(data=model_scores,x='Name',y='Importance');

    


