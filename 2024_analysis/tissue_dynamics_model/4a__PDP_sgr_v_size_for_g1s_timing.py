#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:26:29 2023

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

from numpy import random
from sklearn import metrics
from sklearn.inspection import permutation_importance, partial_dependence

from numpy import random
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale 

def z_standardize(x):
    return (x - np.nanmean(x))/np.std(x)

df_ = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_.csv',index_col=0)
df_g1s = pd.read_csv('/Users/xies/OneDrive - Stanford/Skin/Mesa et al/MLR model/df_g1s.csv',index_col=0)
df_g1s = df_g1s.drop(columns=['age','G1S_logistic'])

#Trim out G2 cells
df_g1s = df_g1s[df_g1s['time_g1s'] >= 0]


X = df_g1s.drop(columns=['time_g1s'])
y = df_g1s['time_g1s']


#Add interaction effects ?
X['vol*sgr'] = z_standardize(X['sgr'] * X['vol_sm'])

feature_names = X.columns

#%% # Parital correlation prediction on Time2G1S

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

forest = RandomForestRegressor(n_estimators=100)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
forest.fit(X_train,y_train)

top_features = feature_names[forest.feature_importances_.argsort()[::-1]]
print(f'Top importances: {top_features[:5]}')

features = [0, 9, (0, 9)]
PartialDependenceDisplay.from_estimator(forest, X, features)

